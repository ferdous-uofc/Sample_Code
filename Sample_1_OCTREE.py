###########################################################################################
#   A data structure implemented for efficiently calculating ray casting of polygonal faces
#   Copyright (C) 2018 Ferdous Ahmed, Graduate Research Assistant, University of Calgary


from typing import Dict
import numpy as np
import time

# the main list containing all the voxels,
# root id and a count of the total number of
# voxels present
class OctreeNodeList:
    def __init__(self):
        self._total_nodes = 0
        self._root_id = 0
        self.OctreeNodeList: Dict[int, OctreeNode] = {}

# main data structure for storing node information
# in our application we are going to store
# the bounding box and a set of data (face information)
class OctreeNode:

    def __init__(self, id=None, min_xyz=None, max_xyz=None):

        # polygon data object
        self.data = []

        # set the minimum and maximum xyz value of the polygon
        self.min = min_xyz
        self.max = max_xyz

        # index to refer to any node object
        self.id = id

        # parent index is initialized as None type
        self.parent = None

        # children nodes are initialized as None type
        self.FNW = None
        self.FNE = None
        self.FSW = None
        self.FSE = None
        self.BNW = None
        self.BNE = None
        self.BSW = None
        self.BSE = None

        # maximum number of nodes allowed
        self.depth = 4

        # Is it the root node?
        self.isroot = False

        # Is it a leaf node?
        self.isleaf = False

    def contains(self, x, y, z):
        if self.min[0] <= x <= self.max[0] and \
                self.min[1] <= y <= self.max[1] and \
                self.min[2] <= z <= self.max[2]:
            return True
        else:
            return False


class Triangle:

    def __init__(self, x=None, y=None, z=None, face=None):
        self.data = face
        self.x = x
        self.y = y
        self.z = z


# add the polygon triangle information given the node and the data
# input:
# nList: (the list of all voxels). any voxel can be located using nList[index?]
# node: one single voxel that is considered for addition of new data
# data: any data that must be stored in the current voxel. In our case data is the face information
# output:
# nList: updated voxel list
def add_polygon_data_to_node(nList=None, node=None, data=None):
    # if the voxel is not a leaf node
    # then addition is not allowed, addition of new node is only
    # allowed at the leaf nodes
    if not node.isleaf:
        return

    # every voxel can only store a finite number of items
    # before the voxel is split into 8 new voxels
    # if the number of data (in this cases the number of faces)
    # is less than the maximum capacity then the data (face) is
    # simply added to the current voxel
    if len(node.data) < node.depth:
        node.data.append(data)

    # if the voxel exceeds the maximum capacity then
    # the current voxel is split into 8 new voxels
    # all the data from current voxel is transferred from
    # current voxel to one of the new voxel, the data of the current
    # voxel is cleared out, finally adjust pointers so that the
    # new voxels can be located later during the traversal
    else:
        # current voxel's status of a leaf node must be removed
        # as new voxels will be attached with this node
        node.isleaf = False

        _total_nodes = nList._total_nodes

        # adding the data to the list before the current voxel's list
        # remember after this function the current voxel's list will become
        # empty and the list data will be transferred to one of the
        # newly created child voxel
        node_list = node.data
        node_list.append(data)

        xmin = node.min[0]
        ymin = node.min[1]
        zmin = node.min[2]

        xmax = node.max[0]
        ymax = node.max[1]
        zmax = node.max[2]

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        # creation of new voxels
        node_FNW = OctreeNode(_total_nodes + 1, [xmin, ymid, zmin], [xmid, ymax, zmid])
        node_FNE = OctreeNode(_total_nodes + 2, [xmid, ymid, zmin], [xmax, ymax, zmid])
        node_FSW = OctreeNode(_total_nodes + 3, [xmin, ymin, zmin], [xmid, ymid, zmid])
        node_FSE = OctreeNode(_total_nodes + 4, [xmid, ymin, zmin], [xmax, ymid, zmid])
        node_BNW = OctreeNode(_total_nodes + 5, [xmin, ymid, zmid], [xmid, ymax, zmax])
        node_BNE = OctreeNode(_total_nodes + 6, [xmid, ymid, zmid], [xmax, ymax, zmax])
        node_BSW = OctreeNode(_total_nodes + 7, [xmin, ymin, zmid], [xmid, ymid, zmax])
        node_BSE = OctreeNode(_total_nodes + 8, [xmid, ymin, zmid], [xmax, ymid, zmax])

        # all new voxel should point to the current voxel as their parent
        node_FNW.parent = node.id
        node_FNE.parent = node.id
        node_FSW.parent = node.id
        node_FSE.parent = node.id
        node_BNW.parent = node.id
        node_BNE.parent = node.id
        node_BSW.parent = node.id
        node_BSE.parent = node.id

        # the id of the new voxels are set in an increasing order
        node.FNW = _total_nodes + 1
        node.FNE = _total_nodes + 2
        node.FSW = _total_nodes + 3
        node.FSE = _total_nodes + 4
        node.BNW = _total_nodes + 5
        node.BNE = _total_nodes + 6
        node.BSW = _total_nodes + 7
        node.BSE = _total_nodes + 8

        # updating the overall voxel count
        # important when we need to generate new voxels and their ids
        nList._total_nodes = nList._total_nodes + 9

        # all the new voxels are set to leaf voxel
        node_FNW.isleaf = True
        node_FNE.isleaf = True
        node_FSW.isleaf = True
        node_FSE.isleaf = True
        node_BNW.isleaf = True
        node_BNE.isleaf = True
        node_BSW.isleaf = True
        node_BSE.isleaf = True

        # transferring the data to one of the child voxel
        for d in node_list:
            if node_FNW.contains(d.x, d.y, d.z):
                node_FNW.data.append(d)
            elif node_FNE.contains(d.x, d.y, d.z):
                node_FNE.data.append(d)
            elif node_FSW.contains(d.x, d.y, d.z):
                node_FSW.data.append(d)
            elif node_FSE.contains(d.x, d.y, d.z):
                node_FSE.data.append(d)
            elif node_BNW.contains(d.x, d.y, d.z):
                node_BNW.data.append(d)
            elif node_BNE.contains(d.x, d.y, d.z):
                node_BNE.data.append(d)
            elif node_BSW.contains(d.x, d.y, d.z):
                node_BSW.data.append(d)
            elif node_BSE.contains(d.x, d.y, d.z):
                node_BSE.data.append(d)

        # clearing the data of the current voxel
        node.data.clear()

        # adjusting the pointers to locate new child voxels
        nList.OctreeNodeList[node.id] = node
        nList.OctreeNodeList[node_FNW.id] = node_FNW
        nList.OctreeNodeList[node_FNE.id] = node_FNE
        nList.OctreeNodeList[node_FSW.id] = node_FSW
        nList.OctreeNodeList[node_FSE.id] = node_FSE
        nList.OctreeNodeList[node_BNW.id] = node_BNW
        nList.OctreeNodeList[node_BNE.id] = node_BNE
        nList.OctreeNodeList[node_BSW.id] = node_BSW
        nList.OctreeNodeList[node_BSE.id] = node_BSE

        return nList

# main function that adds a data (in our case face information) to the right voxel
def add_polygon_data(nList=None, t=None):
    OctreeNodeList = nList.OctreeNodeList
    iter = nList._root_id
    while not OctreeNodeList[iter].isleaf:
        node = OctreeNodeList[iter]
        if OctreeNodeList[node.FNW].contains(t.x, t.y, t.z):
            iter = node.FNW
        elif OctreeNodeList[node.FNE].contains(t.x, t.y, t.z):
            iter = node.FNE
        elif OctreeNodeList[node.FSW].contains(t.x, t.y, t.z):
            iter = node.FSW
        elif OctreeNodeList[node.FSE].contains(t.x, t.y, t.z):
            iter = node.FSE
        elif OctreeNodeList[node.BNW].contains(t.x, t.y, t.z):
            iter = node.BNW
        elif OctreeNodeList[node.BNE].contains(t.x, t.y, t.z):
            iter = node.BNE
        elif OctreeNodeList[node.BSW].contains(t.x, t.y, t.z):
            iter = node.BSW
        elif OctreeNodeList[node.BSE].contains(t.x, t.y, t.z):
            iter = node.BSE

    nList = add_polygon_data_to_node(nList, nList.OctreeNodeList[iter], t)
    return nList

# given the node list (list of current nodes residing in OcTree data structure)
# and a node (triangle) find the voxel containing the triangle
# input:
# nList: (the list of all voxels). any voxel can be located using nList[index?]
# t: any data that must be stored in the current voxel. In our case data is the face information
# output:
# iter: index pointing to correct voxel containing the face
def find_node(nList=None, t=None):
    OctreeNodeList = nList.OctreeNodeList
    iter = nList._root_id

    # return if even the root node doesn't contain the face
    if not OctreeNodeList[iter].contains(t.x, t.y, t.z):
        return -1

    # iterate until the desired voxel containing the face is found
    while not OctreeNodeList[iter].isleaf:
        node = OctreeNodeList[iter]
        if OctreeNodeList[node.FNW].contains(t.x, t.y, t.z):
            iter = node.FNW
        elif OctreeNodeList[node.FNE].contains(t.x, t.y, t.z):
            iter = node.FNE
        elif OctreeNodeList[node.FSW].contains(t.x, t.y, t.z):
            iter = node.FSW
        elif OctreeNodeList[node.FSE].contains(t.x, t.y, t.z):
            iter = node.FSE
        elif OctreeNodeList[node.BNW].contains(t.x, t.y, t.z):
            iter = node.BNW
        elif OctreeNodeList[node.BNE].contains(t.x, t.y, t.z):
            iter = node.BNE
        elif OctreeNodeList[node.BSW].contains(t.x, t.y, t.z):
            iter = node.BSW
        elif OctreeNodeList[node.BSE].contains(t.x, t.y, t.z):
            iter = node.BSE

    return iter
