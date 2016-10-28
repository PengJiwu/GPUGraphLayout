/*
 ==============================================================================
 
 main.cpp
 
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
#include <string>
#include <math.h>

#include "../lib/snap/snap-core/Snap.h"

#include "RPCommon.hpp"
#include "RPGraph.hpp"
#include "RPGraphLayout.hpp"
#include "RPForceAtlas2.hpp"

#ifdef __CUDA__
#include <cuda_runtime_api.h>
#include "RPCUDAForceAtlas2.hpp"
#endif

int main(int argc, const char **argv)
{
    // For reproducibility.
    srandom(1234);
    
    // Parse commandline arguments
    if (argc != 12)
    {
        fprintf(stderr, "Usage: graph_viewer cuda|seq gc|all max_iterations num_snaps sg|wg scale gravity exact|approximate edgelist_path out_path test|image\n");
        exit(EXIT_FAILURE);
    }
    
    const bool cuda_requested = std::string(argv[1]) == "cuda";
    const bool filter_gc = std::string(argv[2]) == "gc";
    const int max_iterations = std::stoi(argv[3]);
    const int num_screenshots = std::stoi(argv[4]);
    const bool strong_gravity = std::string(argv[5]) == "sg";
    const float scale = std::stof(argv[6]);
    const float gravity = std::stof(argv[7]);
    const bool approximate = std::string(argv[8]) == "approximate";
    const char *edgelist_path = argv[9];
    const char *out_path = argv[10];
    const bool testmode = std::string(argv[11]) == "test";
    const int framesize = 10000;
    const float w = framesize;
    const float h = framesize;
    
    // Check in_path and out_path
    if (!is_file_exists(edgelist_path))
    {
        fprintf(stderr, "error: No edgelist at %s\n", edgelist_path);
        exit(EXIT_FAILURE);
    }
    if (!is_file_exists(out_path))
    {
        fprintf(stderr, "error: No output folder at %s\n", out_path);
        exit(EXIT_FAILURE);
    }
    
    printf("Loading edgelist at '%s'...", edgelist_path);
    fflush(stdout);
    PUNGraph graph_snap = TSnap::LoadEdgeList<PUNGraph>(edgelist_path);
    printf("done.\n");
    printf("    fetched %d nodes and %d edges.\n", graph_snap->GetNodes(), graph_snap->GetEdges());
    
    if (filter_gc)
    {
        printf("Filtering for giant component...");
        fflush(stdout);
        graph_snap = TSnap::GetMxScc(graph_snap);
        printf("done.\n");
        printf("    %d nodes and %d edges remain.\n", graph_snap->GetNodes(), graph_snap->GetEdges());
    }
    
    if (TSnap::CntSelfEdges(graph_snap) > 0)
    {
        printf("Removing self-edges...");
        fflush(stdout);
        TSnap::DelSelfEdges(graph_snap);
        printf("done.\n");
        printf("    removed %d self-edges.\n", TSnap::CntSelfEdges(graph_snap));
    }
    
    printf("Converting to CSR format...");
    fflush(stdout);
    RPGraph::CSRUGraph graph = RPGraph::CSRUGraph(graph_snap->GetNodes(), graph_snap->GetEdges());
    for (TUNGraph::TNodeI NI = graph_snap->BegNI(); NI < graph_snap->EndNI(); NI++)
    {
        if (NI.GetDeg() == 0) continue;
        RPGraph::nid_t nid = NI.GetId();
        std::vector<RPGraph::nid_t> eids;
        
        for (int ei = 0; ei < NI.GetDeg(); ei++)
        {
            eids.push_back(NI.GetNbrNId(ei));
        }
        graph.insert_node(nid, eids);
    }
    graph.fix_edge_ids();
    printf("done.\n");
    
    if(cuda_requested)
    {
#ifdef __CUDA__
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            fprintf(stderr, "error: There is no device supporting CUDA...\n");
            exit(EXIT_FAILURE);
        }
        
        RPGraph::CUDAFA2Layout layout(graph, w, h);
        layout.strong_gravity = strong_gravity;
        layout.use_barneshut = approximate;
        layout.setScale(scale);
        layout.setGravity(gravity);
        
        if(testmode) {
	        layout.benchmark();
	        exit(EXIT_SUCCESS);
	    }
	    
        printf("Started Layout algorithm...\n");
        const int snap_period = ceil((float)max_iterations/num_screenshots);
        const int print_period = ceil((float)max_iterations*0.05);
        
        for (int iteration = 1; iteration <= max_iterations; ++iteration)
        {
            layout.doStep();
            // If we need to, write the result to a png
            if (num_screenshots > 0 && (iteration % snap_period == 0 || iteration == max_iterations))
            {
                std::string op(out_path);
                op.append("/").append(std::to_string(iteration)).append(".png");
                printf("Starting iteration %d (%.2f%%), writing png...", iteration, 100*(float)iteration/max_iterations);
                fflush(stdout);
                layout.writeToPNG(framesize, framesize, op.c_str());
                printf("done.\n");
            }
            
            // Else we print (if we need to)
            else if (iteration % print_period == 0)
            {
                printf("Starting iteration %d (%.2f%%).\n", iteration, 100*(float)iteration/max_iterations);
            }
        }
        printf("Done with the layout process, writing layout to file...");
        fflush(stdout);
        std::string op(out_path);
//        layout.writePositions(op.append("layout.txt").c_str());
        printf("done.\n");
        
#else
        fprintf(stderr, "error: CUDA was requested, but not compiled for.\n");
        exit(EXIT_FAILURE);
#endif
    }
    
    else if(!cuda_requested)
    {
        // Create the layout
        RPGraph::FA2Layout layout = RPGraph::FA2Layout(graph, w, h);
        layout.strong_gravity = strong_gravity;
        layout.use_barneshut = approximate;
        layout.setScale(scale);
        layout.setGravity(gravity);

        if(testmode)
        {
            layout.doSteps(100);
            layout.print_benchmarks();
            exit(EXIT_SUCCESS);
        }
        
        printf("Started Layout algorithm...\n");
        const int snap_period = ceil((float)max_iterations/num_screenshots);
        const int print_period = ceil((float)max_iterations*0.05);
        for (int iteration = 1; iteration <= max_iterations; ++iteration)
        {
            layout.doStep();
            // If we need to, write the result to a png
            if (num_screenshots > 0 && (iteration % snap_period == 0 || iteration == max_iterations))
            {
                std::string op(out_path);
                op.append("/").append(std::to_string(iteration)).append(".png");
                printf("At iteration %d (%.2f%%), writing png...", iteration, 100*(float)iteration/max_iterations);
                fflush(stdout);
                layout.writeToPNG(framesize, framesize, op.c_str());
                printf("done.\n");
            }
            
            // Else we print (if we need to)
            else if (iteration % print_period == 0)
            {
                printf("At iteration %d (%.2f%%).\n", iteration, 100*(float)iteration/max_iterations);
            }
        }
//        printf("Done with the layout process, writing layout to file...");
        fflush(stdout);
        std::string op(out_path);
//        layout.writePositions(op.append("layout.txt").c_str());
        printf("done.\n");
    }
    exit(EXIT_SUCCESS);
}
