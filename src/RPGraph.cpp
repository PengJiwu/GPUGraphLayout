/*
 ==============================================================================
 
 RPGraph.cpp
 
 This code was written as part of a research project at the Leiden Institute of
 Advanced Computer Science (www.liacs.nl). For other resources related to this
 project, see http://liacs.leidenuniv.nl/~takesfw/GPUNetworkVis/.
 
 Copyright (C) 2016  G. Brinkmann
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 ==============================================================================
*/


#include <stdio.h>
#include <stdlib.h>
#include "RPGraph.hpp"

namespace RPGraph
{
    /* Definitions for CSRUGraph */
    
// CSRUGraph represents an undirected graph using a
// compressed sparse row (CSR) datastructure.
    CSRUGraph::CSRUGraph(nid_t num_nodes, nid_t num_edges)
    {
        // `edges' is a concatenation of all edgelists
        // `offsets' contains offset (in `edges`) for each nodes' edgelist.
        // `nid_to_offset` maps nid to index to be used in `offset'
        
        // e.g. the edgelist of node with id `nid' starts at
        // edges[offsets[nid_to_offset[nid]]] and ends at edges[offsets[nid_to_offset[nid]] + 1]
        // (left bound inclusive right bound exclusive)
        
        edge_count = num_edges; // num_edges counts each bi-directional edge once.
        node_count = num_nodes;
        edges =   (nid_t *) malloc(sizeof(nid_t) * 2 * edge_count);
        offsets = (nid_t *) malloc(sizeof(nid_t) * node_count);
        offset_to_nid = (nid_t *) malloc(sizeof(nid_t) * node_count);
        
        // Create a map from original ids to ids used throughout CSRUGraph
        nid_to_offset.reserve(node_count);
        
        first_free_id = 0;
        edges_seen = 0;
    }
    
    CSRUGraph::~CSRUGraph()
    {
        free(edges);
        free(offsets);
        free(offset_to_nid);
    }
        
    void CSRUGraph::insert_node(nid_t node_id, std::vector<nid_t> nbr_ids)
    {
        nid_t source_id_old = node_id;
        nid_t source_id_new = first_free_id;
        nid_to_offset[source_id_old] = first_free_id;
        offset_to_nid[first_free_id] = source_id_old;
        first_free_id++;
        
        offsets[source_id_new] = edges_seen;
        for (auto nbr_id : nbr_ids)
        {
            nid_t dest_id_old = nbr_id;
            edges[edges_seen] = dest_id_old;
            edges_seen++;
        }
    }
    
    void CSRUGraph::fix_edge_ids()
    {
        for (eid_t ei = 0; ei < 2*edge_count; ei++)
        {
            edges[ei] = nid_to_offset[edges[ei]];
        }
    }
    
    nid_t CSRUGraph::degree(nid_t nid)
    {
        // If nid is last element of `offsets'... we prevent out of bounds.
        nid_t r_bound;
        if (nid < node_count - 1) r_bound = offsets[nid+1];
        else r_bound = edge_count * 2;
        nid_t l_bound = offsets[nid];
        return (r_bound - l_bound);
    }
    
    nid_t CSRUGraph::out_degree(nid_t nid)
    {
        return degree(nid);
    }
    
    nid_t CSRUGraph::in_degree(nid_t nid)
    {
        return degree(nid);
    }
    
    nid_t CSRUGraph::nbr_id_for_node(nid_t nid, nid_t edge_no)
    {
        return edges[offsets[nid] + edge_no];
    }
    nid_t CSRUGraph::num_nodes()
    {
        return node_count;
    }

    nid_t CSRUGraph::num_edges()
    {
        return edge_count;
    }
};