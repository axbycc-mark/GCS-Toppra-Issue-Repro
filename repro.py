from collections import defaultdict
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.csgraph import dijkstra
from types import SimpleNamespace
import itertools
import numpy as np
import os
import pickle
import random
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AddDefaultVisualization,
    Box,
    CollisionFilterDeclaration,
    CoulombFriction,
    CompositeTrajectory,
    CalcGridPointsOptions,
    DiagramBuilder,
    FixedOffsetFrame,
    GcsTrajectoryOptimization,
    GeometrySet,
    HPolyhedron,
    InverseKinematics,
    Iris,
    IrisInConfigurationSpace,
    IrisOptions,
    JacobianWrtVariable,
    LoadIrisRegionsYamlFile,
    MathematicalProgram,
    MeshcatPoseSliders,
    Parser,
    Point,
    QueryObject,
    RandomGenerator,
    Rgba,
    RigidTransform,
    PiecewisePolynomial,
    PathParameterizedTrajectory,
    Role,
    RollPitchYaw,
    RotationMatrix,
    SaveIrisRegionsYamlFile,
    Solve,
    Sphere,
    StartMeshcat,
    Toppra,
    ToppraDiscretization
)

def add_xarm7(plant):
    xarm7_xoffset = 0.1
    xarm7_yoffset = 0.1
    
    mounting_plate_thickness = 0.00823
    mounting_plate_short_side = 0.136
    mounting_plate_long_side = 0.180
    mounting_plate_position = np.asarray(
        [
            mounting_plate_short_side/2 + 0.035,
            mounting_plate_long_side/2 + 0.015,
            mounting_plate_thickness/2,
        ]
    )
    mounting_plate = Box(depth=mounting_plate_long_side,
                         width=mounting_plate_short_side,
                         height=mounting_plate_thickness)
    mounting_plate_color = np.asarray([0.5, 0.5, 0.5, 1.0])
    plant.RegisterVisualGeometry(
        plant.world_body(), RigidTransform(p=mounting_plate_position),
        mounting_plate, "MountingPlateVisualGeometry",
        mounting_plate_color)
    plant.RegisterCollisionGeometry(
        plant.world_body(), RigidTransform(p=mounting_plate_position),
        mounting_plate, "MountingPlateCollisionGeometry",
        CoulombFriction(1.0, 1.0))
    
    parser = Parser(plant)
    xarm7 = parser.AddModels("xarm_description/xarm7.urdf")[0]
    xarm_base = plant.GetBodyByName("link_base", xarm7)
    plant.WeldFrames(plant.world_frame(),
                     xarm_base.body_frame(),
                     RigidTransform(
                         RollPitchYaw(np.asarray([0.0, 0.0, np.pi])),
                         p=np.asarray([
                             xarm7_xoffset,
                             xarm7_yoffset,
                             mounting_plate_thickness])))

    return xarm7

class TrajectoryPlanner:
    def __init__(self, diagram, iris_regions, save_path=None):
        self.diagram = diagram
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.dofs = self.plant.num_positions()
        self.velocity_ub = self.plant.GetVelocityUpperLimits()
        self.velocity_lb = self.plant.GetVelocityLowerLimits()
        self.accel_ub = self.plant.GetAccelerationUpperLimits()
        self.accel_lb = self.plant.GetAccelerationLowerLimits()
        self.iris_regions = iris_regions
        self.iris_portals = IrisPortals(iris_regions, save_path)

    def choose_random_q(self, drake_rng):
        random_region = self.iris_portals.region_values[random.randint(0, self.iris_portals.num_regions-1)]
        q = random_region.UniformSample(drake_rng, mixing_steps=1000)
        return q

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=0)
    add_xarm7(plant)
    plant.Finalize()
    diagram = builder.Build()
    dofs = plant.num_positions()
    gcs = GcsTrajectoryOptimization(num_positions=dofs)

    # the path through portals implies a region visit order
    # collect the ordered sequence of regions visited
    region = HPolyhedron.MakeBox(np.zeros(dofs), np.ones(dofs))

    starting_q = np.array(np.ones(dofs)) * 0.1
    ending_q = np.array(np.ones(dofs)) * 0.9
    gcs_regions = [Point(starting_q), region, Point(ending_q)]

    first_gcs_region = gcs_regions[1]
    assert first_gcs_region.PointInSet(starting_q)
    last_gcs_region = gcs_regions[-2]
    assert last_gcs_region.PointInSet(ending_q)

    gcs_edges = []
    for i in range(len(gcs_regions)-1):
        gcs_edges.append((i, i+1))

    for i,j in gcs_edges:
        assert gcs_regions[i].IntersectsWith(gcs_regions[j])

    # add regions for start and end
    subgraph = gcs.AddRegions(gcs_regions,
                              gcs_edges,
                              order=3)
    vertices = subgraph.Vertices()
    gcs.AddTimeCost()
    gcs.AddVelocityBounds(plant.GetVelocityLowerLimits(),
                          plant.GetVelocityUpperLimits())
    gcs.AddPathLengthCost(1.0)
    gcs.AddPathContinuityConstraints(1)
    # gcs.AddPathContinuityConstraints(2)
    # gcs.AddPathContinuityConstraints(3)


    print("Running GCS")
    gcs_traj, result = gcs.SolveConvexRestriction(vertices)
    if not result.is_success():
        print("Solution failed. Dumping graphviz to gcs.txt")
        solution_result = result.get_solution_result()
        graph_viz = gcs.GetGraphvizString()
        with open("gcs.txt", "w") as f:
            f.write(graph_viz)
        raise RuntimeError(f"GCS Failed {solution_result}")

    gcs_traj = gcs.NormalizeSegmentTimes(gcs_traj)

    # set to True to display a plot of joints vs time
    plot_gcs_traj = False
    if plot_gcs_traj:
        ts = np.linspace(gcs_traj.start_time(), gcs_traj.end_time(), 1000)
        gcs_traj_points = []
        for t in ts:
            gcs_traj_points.append(gcs_traj.value(t).flatten())

        gcs_traj_points = np.array(gcs_traj_points)
        for i in range(dofs):
            plt.plot(ts, gcs_traj_points[:,i])
        plt.show()

    # clip stationary start and end segments for toppra
    # set to True to fix the Toppra failure
    clip_terminal_segments = False
    if clip_terminal_segments:
        num_gcs_segments = gcs_traj.get_number_of_segments()        
        gcs_clip_reparam = PiecewisePolynomial.FirstOrderHold(
            np.array([0.0, 1.0]),
            np.array([1.0, float(num_gcs_segments-1)]).reshape((1,2))
        )
        gcs_traj = PathParameterizedTrajectory(gcs_traj, gcs_clip_reparam)

    grid_points = Toppra.CalcGridPoints(gcs_traj, CalcGridPointsOptions())
    toppra = Toppra(gcs_traj, plant, grid_points)
    toppra.AddJointVelocityLimit(plant.GetVelocityLowerLimits(),
                                 plant.GetVelocityUpperLimits())

    # xarm urdf does not provide acceleration limits
    # supplying dummy ones here
    accel = 10
    accel_lb = -np.ones(dofs) * accel
    accel_ub = np.ones(dofs) * accel
    toppra.AddJointAccelerationLimit(accel_lb, accel_ub)        

    print("Running Toppra")
    toppra_times = toppra.SolvePathParameterization()

    if toppra_times is None:
        raise RuntimeError("Toppra failed")

    return toppra_times, gcs_traj


    
if __name__ == "__main__":
    import pydrake
    print("Solver infos")
    print(pydrake.solvers.MakeFirstAvailableSolver([
        pydrake.solvers.ClpSolver.id(), pydrake.solvers.MosekSolver.id()
    ]).solver_id().name())
    main()
