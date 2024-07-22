
from collections import defaultdict
import numpy as np

class TriNeighbors:
    neighbors = None
    def __init__(self, t):
        elem2edge, edge2elem = self.get_edge_elem_relationship(t)
        # keep nearest neighbors for each element
        self.neighbors = []
        for i in range(t.shape[0]):
            neighbors_lists = [edge2elem[edge] for edge in elem2edge[i] if edge in edge2elem]
            neighbors_i = []
            for n1 in neighbors_lists:
                if len(n1) != 1:
                    for n2 in n1:
                        if n2 != i:
                            neighbors_i.append(n2)
            self.neighbors.append(neighbors_i)
        self.nneighbors = [len(n) for n in self.neighbors]
        
    def get_edge_elem_relationship(self, t):
        """ Create to maps: element to edge, and edge to element in one pass"""
        elem2edge = []
        edge2elem = defaultdict(list)
        for i, elem in enumerate(t):
            jj = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
            edges = [tuple(sorted(j)) for j in jj]
            elem2edge.append(edges)
            for edge in edges:
                edge2elem[edge].append(i)
        return elem2edge, edge2elem
    
    def get_neighbors(self, i, n=1, e=()):
        """ Get neighbors of element <i> crossing <n> edges

        Parameters
        ----------
        i : int, index of element
        n : int, number of edges to cross
        e : (int,), indeces to exclude

        Returns
        -------
        l : list, List of unique inidices of existing neighbor elements

        """
        # return list of existing immediate neigbours
        if n == 1:
            return self.neighbors[i]
        # recursively return list of neighbors after 1 edge crossing
        n2 = []
        for j in self.neighbors[i]:
            if j not in e:
                n2.extend(self.get_neighbors(j, n-1, e+(i,)))
        return list(set(self.neighbors[i] + n2))

    def get_neighbors_many(self, indices, n=1):
        """ Group neighbours of several elements """
        neighbors_many = [self.get_neighbors(i, n=n) for i in indices]
        return np.unique(np.hstack(neighbors_many)).astype(int)
    
    def get_distance_to_border(self):
        dist = np.zeros(len(self.neighbors)) + np.nan
        border = np.where(np.array(self.nneighbors) < 3)[0]
        dist[border] = 0

        d = 1
        while np.any(np.isnan(dist)):
            for i in np.where(dist == d - 1)[0]:
                neibs = self.get_neighbors(i)
                for j in neibs:
                    if np.isnan(dist[j]):
                        dist[j] = d
            d += 1
        return dist