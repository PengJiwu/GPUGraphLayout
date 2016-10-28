/*
 ==============================================================================
 
 RPGraph.hpp
 
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


#ifndef RPGraph_hpp
#define RPGraph_hpp
#include <unordered_map>
#include <vector>

namespace RPGraph
{
    // Type to represent node IDs.
    // NOTE: we limit to 4,294,967,296 nodes through uint32_t.
    typedef uint32_t nid_t;
    
    // Type to represent edge IDs.
    typedef uint32_t eid_t;
    
    // Virtual base class to derive different Gaphs types from.
    class Graph
    {
        public:
            virtual nid_t num_nodes() = 0;
            virtual nid_t num_edges() = 0;
            virtual nid_t degree(nid_t nid) = 0;
            virtual nid_t in_degree(nid_t nid) = 0;
            virtual nid_t out_degree(nid_t nid) = 0;
    };
    
    // Compressed sparserow (CSR) for undirected graphs.
    class CSRUGraph : public Graph
    {
    private:
        nid_t *edges;   // All edgelists, concatenated.
        nid_t *offsets; // For each node, into edges.
        nid_t node_count, edge_count;
        nid_t first_free_id, edges_seen;
        
    public:
        std::unordered_map<nid_t, nid_t> nid_to_offset;
        nid_t *offset_to_nid;

        CSRUGraph(nid_t num_nodes, nid_t num_edges);
        ~CSRUGraph();
        
        /// Inserts node_id and its edges. Once inserted, edges
        /// can't be altered for this node.
        void insert_node(nid_t node_id, std::vector<nid_t> nbr_ids);
        void fix_edge_ids(); // this should go...

        virtual nid_t num_nodes() override;
        virtual nid_t num_edges() override;
        virtual nid_t degree(nid_t nid) override;
        virtual nid_t in_degree(nid_t nid) override;
        virtual nid_t out_degree(nid_t nid) override;

        nid_t nbr_id_for_node(nid_t nid, nid_t nbr_no);
    };
}

#endif /* Graph_h */
