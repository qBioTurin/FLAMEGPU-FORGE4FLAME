#!/usr/bin/env python3
import numpy as np
import pandas as pd
import logging
import math
from MapEncoding import *
from typing import List, Set, Union
from itertools import chain
from bresenham import bresenham
from collections import deque

first_vertex_id = 0

class Coordinates:
    def __init__(self, x, y, z, center_x, center_z, nw, se) -> None:
        self.__point = np.array([x, y, z])
        self.__center_x = center_x
        self.__center_z = center_z
        self.__nw_corder = nw
        self.__se_corner = se
    
    @property
    def vec(self):
        return self.__point

    @property
    def x(self):
        return self.__point[0]
    
    @property
    def y(self):
        return self.__point[1]
    
    @property
    def z(self):
        return self.__point[2]
    
    @property
    def center_x(self):
        return self.__center_x

    @property
    def center_z(self):
        return self.__center_z

    @property
    def northwest(self):
        return self.__nw_corder
    
    @property
    def southeast(self):
        return self.__se_corner
    
    def distance(self, other) -> Union[int, float]:
        assert isinstance(other, Coordinates)
        return np.sum(np.abs(self.__point - other.__point))


class Vertex:
    """ A vertex is identified by a set of coordinates (x, y, z) and a label representing its type """

    def __init__(self, vid: int, 
                 coordinates: Coordinates,
                 typeof: MapEncoding,
                 area: int,
                 yaw: float,
                 length: int,
                 width: int,
                 resources: pd.DataFrame,
                 waitingroom_det: pd.DataFrame,
                 waitingroom_rand: pd.DataFrame,
                 room_matrix_withmask: np.ndarray,
                 room_matrix_withoutmask: np.ndarray,
                 objects: pd.DataFrame,
                 doorsINcanvas: pd.DataFrame,
                 graph_id: int,
                 room_container_id: int,
                 door_container_id: int,
                 f4f_id: int) -> None:
        self.id = vid 
        self.coords = coordinates
        self.type = typeof
        self.area = area
        self.yaw = yaw
        self.length = length
        self.width = width
        self.resources = resources
        self.waitingroom_det = waitingroom_det
        self.waitingroom_rand = waitingroom_rand
        self.room_matrix_withmask = room_matrix_withmask
        self.room_matrix_withoutmask = room_matrix_withoutmask
        self.objects = objects
        self.doorsINcanvas = doorsINcanvas
        self.graph_id = graph_id
        self.room_container_id = room_container_id
        self.door_container_id = door_container_id
        self.f4f_id = f4f_id

    def __str__(self):
        return f"{self.id} {MapEncoding.to_str(self.type)} {int(self.coords.x)} {int(self.coords.y)} {int(self.coords.z)}"

    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Vertex) and self.id == __value.id

    def __hash__(self) -> int:
        return hash(self.id)


class GraphEdge:
    """ An edge connects two vertices and it is weighted as the distance between them  """

    def __init__(self, v1: Vertex, v2: Vertex, distance = None): 
        self.v1, self.v2 = (v1, v2) if v1.id < v2.id else (v2, v1)

        if distance == None:
            self.w = int(v1.coords.distance(v2.coords)) # abs(v1.x - v2.x) + abs(v1.y - v2.y) + abs(v1.z - v2.z)
        else:
            self.w = distance

    def __str__(self) -> str:
        return f"{self.v1.id} {self.v2.id} {self.w}"

    def __repr__(self) -> str:
        return f"Edge from {self.v1} to {self.v2}"
    
    def __hash__(self) -> int:
        return (self.v1.id, self.v2.id).__hash__()
    
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, GraphEdge) and self.v1 == __value.v1 and self.v2 == __value.v2 


class SpatialGraph:
    """ A spatial graph is a representation of a map using an undirected labeled graph, where 
     vertices are identified by (x, y, z) coordinates and are labeled with the type of cell, 
     and edges are weighted as the Manhattan distance between the two vertices, which 
     are on the same horizontal/vertical line  """

    def __init__(self) -> None:
        global first_vertex_id
        self.vertices = dict()
        self.edgelist = set()
        self.__first_vid = first_vertex_id

        for vtype in MapEncoding:
            if vtype not in [MapEncoding.CORRIDOR, MapEncoding.INSIDEROOM]:
                self.vertices[vtype] = []

    def add_vertex(self, x: int, y: int, z: int, center_x: int, center_z: int, northwest: list, southeast: list, vtype: MapEncoding, area: int, yaw: float, length: int, width: int, resources: pd.DataFrame, waitingrooms_det: pd.DataFrame, waitingrooms_rand: pd.DataFrame, room_matrix_withmask: np.ndarray, room_matrix_withoutmask: np.ndarray, objects: pd.DataFrame, doorsINcanvas: pd.DataFrame, graph_id: int, room_container_id: int, door_container_id: int, f4f_id: int):
        global first_vertex_id

        self.vertices[vtype].append(Vertex(self.__first_vid, Coordinates(x, y, z, center_x, center_z, northwest, southeast), vtype, area, yaw, length, width, resources, waitingrooms_det, waitingrooms_rand, room_matrix_withmask, room_matrix_withoutmask, objects, doorsINcanvas, graph_id, room_container_id, door_container_id, f4f_id))
        self.__first_vid = self.__first_vid + 1
        first_vertex_id = first_vertex_id + 1

    def init_edges(self, mapsource: Union[ str, np.ndarray ]):
        if mapsource is not None:
            if isinstance(mapsource, str):
                with open(mapsource) as f:
                    # Build a list of np.arrays using lines read from the input file 
                    int_lines = [
                        np.array([ int(char) for char in line.split(",") ])
                            for line in f 
                    ]

                    # Build a matrix using each line as a row 
                    self.matrix = np.vstack(int_lines).T
                    
                    logging.info(f"Map shape: {self.matrix.shape}")
            elif isinstance(mapsource, np.ndarray):
                self.matrix = mapsource
            else:
                raise RuntimeError("Please provide either a filename or a numpy matrix as input parameter.")

        for me in MapEncoding:
            if me not in [MapEncoding.CORRIDOR, MapEncoding.INSIDEROOM, MapEncoding.DOOR, MapEncoding.CPOINT, MapEncoding.FILLINGROOM]:
                self.edgelist.update(self.__match_doors(MapEncoding.DOOR, me))

        self.edgelist.update(self.__match_vertices(MapEncoding.DOOR, MapEncoding.DOOR))
        self.edgelist.update(self.__match_vertices(MapEncoding.CPOINT, MapEncoding.DOOR))
        self.edgelist.update(self.__match_vertices(MapEncoding.CPOINT, MapEncoding.CPOINT))

    @property
    def num_vertices(self):
        return sum([ len(v_list) for v_list in self.vertices.values() ])
    
    @property
    def num_edges(self):
        return len(self.edgelist)
    
    def save(self, filename):
        with open(filename, "w") as f:
            ## GRAPH NAME 
            #  NUMBER OF VERTICES
            f.write(f"{self.num_vertices}\n")

            #  LIST OF VERTICES         
            vlist = sorted(chain.from_iterable(self.vertices.values()), key = lambda v: v.id)
            f.writelines([ f"{v}\n" for v in vlist ])

            #  NUMBER OF EDGES
            f.write(f"{self.num_edges}\n")

            #  LIST OF EDGES 
            sorted_edges = sorted(self.edgelist, key = lambda e: (e.v1.id, e.v2.id))
            f.writelines([ f"{e}\n"  for e in sorted_edges ])

    def __match_vertices(self, type1: MapEncoding, type2: MapEncoding) -> Set[GraphEdge]:
        points_t1 = self.vertices.get(type1)
        points_t2 = self.vertices.get(type2)

        edgelist = {
            GraphEdge(t1, t2, len(path))
            for t1 in points_t1
                for t2 in points_t2
                    if t1 != t2 and (path := self.__check_vertex_compatibility(t1, t2)) and self.__check_middle_values(path, self.matrix)
        }

        return edgelist
    
    def __check_middle_values(self, path, matrix):
        if len(path) <= 2:
            return True

        rows = len(matrix)
        cols = len(matrix[0])

        for x, z in path:
            if not (0 <= x < cols and 0 <= z < rows):
                return False

            value = matrix[z][x]

            # Only 0 and 2 are allowed
            if value not in (0, 2, 3):
                return False

            # 0 is always allowed
            if value in (0, 3):
                continue

            # value == 2: exactly 3 of 4 neighbors must be 0
            neighbors = [
                (x - 1, z),
                (x + 1, z),
                (x, z - 1),
                (x, z + 1)
            ]

            zero_or_two_neighbors = sum(
                0 <= nx < cols
                and 0 <= nz < rows
                and matrix[nz][nx] == 0 or matrix[nz][nx] == 2
                for nx, nz in neighbors
            )

            if zero_or_two_neighbors != 3:
                return False

        return True

    def __match_doors(self, type1: MapEncoding, type2: MapEncoding) -> List[GraphEdge]:
        """ Build edges between two vertices v1 and v2 s.t. type(v1) = type_v1 and type(v2) = type_v2
            and their coordinates are on the same line and without obstacles """
        
        points_t1 = self.vertices.get(type1)
        points_t2 = self.vertices.get(type2)
        bits_vector = [ False ] * len(points_t2)

        edge_list = set()
        for v1 in points_t1:
            for i, v2  in enumerate(points_t2):
                if self.__check_vertex_compatibility_doors(v1, v2):
                    bits_vector[i] = True 
                    edge_list.add(GraphEdge(v1, v2))

        try:
            assert all(bits_vector)
        except AssertionError:
            lv = [ v for v, bit in zip(points_t2, bits_vector) if not bit ]
            logging.warning(f"Unmatched vertices: {lv}")

        logging.info(f"{type1} vs {type2}: {len(edge_list)} edges")
        return edge_list

    def __check_vertex_compatibility_doors(self, v1: Vertex, v2: Vertex) -> bool:
        """ Check if the door is in the same room container as the door considered (belongs to it) """

        return v1.door_container_id == v2.door_container_id

    def __check_vertex_compatibility(self, v1: Vertex, v2: Vertex) -> bool:
        """ Check if two vertices are on the same line (either horizontal or vertical) and there are no wall between them """

        path = list(bresenham(int(v1.coords.x), int(v1.coords.z), int(v2.coords.x), int(v2.coords.z)))

        return path

    def __match_stairs(self):
        stair_vertices = self.vertices.get( MapEncoding.STAIR )
        stair_links = [
            GraphEdge(stair_i, stair_j)

            for i, stair_i in enumerate( stair_vertices )
                for stair_j in stair_vertices[i+1:]
                    if abs(stair_i.coords.x - stair_j.coords.x) <= 5 and abs(stair_i.coords.z - stair_j.coords.z) <= 5
        ]
        self.edgelist.update( stair_links )

    def __bfs_shortest_path(self, start_vertex: Vertex) -> dict:
        """
        Perform BFS from a start vertex and return the shortest path lengths
        to all other vertices.
        """
        distances = {v: float('-inf') for v in chain.from_iterable(self.vertices.values())}

        distances[start_vertex] = 0
        queue = deque([start_vertex])


        while queue:
            current = queue.popleft()

            # For all edges connected to the current vertex
            for edge in self.edgelist:
                if edge.v1 == current and distances[edge.v2] == float('-inf'):
                    distances[edge.v2] = distances[current] + 1
                    queue.append(edge.v2)
                elif edge.v2 == current and distances[edge.v1] == float('-inf'):
                    distances[edge.v1] = distances[current] + 1
                    queue.append(edge.v1)

        return distances

    @classmethod
    def merge_graphs(cls, graph_list: List):
        """ Build a unique graph from a list of spatial graph by building edges between STAIR vertices alignes on (x, z) axes """

        whole_graph = SpatialGraph()
        whole_graph.vertices = { v_type: list() for v_type in MapEncoding }

        for graph in graph_list:
            for v_type, v_list in graph.vertices.items():
                whole_graph.vertices[ v_type ].extend( v_list )
            
            whole_graph.edgelist.update( graph.edgelist )

        del whole_graph.vertices[ MapEncoding.INSIDEROOM ]
        del whole_graph.vertices[ MapEncoding.CORRIDOR ]

        whole_graph.__match_stairs()

        return whole_graph

    def graph_diameter(self) -> int:
        """
        Optimized method to compute graph diameter.
        Uses two BFS passes for efficiency.
        """
        vertices = list(chain.from_iterable(self.vertices.values()))
        
        if not vertices:
            return 0  # No vertices in the graph
        
        # Step 1: Pick an arbitrary vertex and run BFS
        start_vertex = vertices[0]
        shortest_paths = self.__bfs_shortest_path(start_vertex)

        # Step 2: Find the farthest vertex from the first BFS
        farthest_vertex = max(shortest_paths, key=shortest_paths.get)

        # Step 3: Run BFS again from this farthest vertex
        final_shortest_paths = self.__bfs_shortest_path(farthest_vertex)

        # Step 4: Return the longest shortest path found
        return max(final_shortest_paths.values())