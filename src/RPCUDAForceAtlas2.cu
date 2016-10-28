/*
 ==============================================================================
 
 RPCUDAForceAtlas2.cu
 
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
#include <fstream>
#include <chrono>
#include "time.h"

#include "RPCUDAForceAtlas2.hpp"
#include "RPCommon.hpp"

#include "RPCUDALaunchParameters.cuh"
#include "RPBHKernels.cuh"
#include "RPFA2Kernels.cuh"

#include "../lib/pngwriter/src/pngwriter.h"

namespace RPGraph
{
    CUDAFA2Layout::CUDAFA2Layout(CSRUGraph &graph, float width, float height)
    : graph(graph), width(width), height(height)
    {
        /*** General FA2 code***/
        iteration = 0;
        
        k_g = 1.0;
        k_r = 1.0;
        
        global_speed = 1.0;
        speed_efficiency = 1.0;
        jitter_tolerance = 1.0;
        
        k_s = 0.1;
        k_s_max = 10.0;
        theta = 1.0;
        
        delta = 0.0;
        
        prevent_overlap = false;
        strong_gravity = false;
        dissuade_hubs = false;
        use_barneshut = true;
        use_linlog = false;
        
        if (!use_barneshut)
        {
            printf("RPCUDAForeceAtlas2 without Barnes-Hut approximation is not implemented yet.\n");
            exit(EXIT_FAILURE);
        }

        /*** CUDA Specific code ***/
        // Determine number of multiprocessort on GPU.
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        mp_count = deviceProp.multiProcessorCount;
        int warpSize = deviceProp.warpSize;
        if (warpSize != WARPSIZE)
        {
            printf("Warpsize of device is %d, but we anticipated %d\n", warpSize, WARPSIZE);
            exit(EXIT_FAILURE);
            
        }
        cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
#if __CUDA_ARCH__ < 300
        cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
#endif
        cudaFuncSetCacheConfig(DisplacementKernel, cudaFuncCachePreferL1);
        
        cudaGetLastError();  // reset error value
        
        
        // Allocate space on host.
        nbodies = graph.num_nodes();
        nedges  = graph.num_edges();

        posx     = (float *)malloc(sizeof(float) * graph.num_nodes());
        posy     = (float *)malloc(sizeof(float) * graph.num_nodes());
        mass     = (float *)malloc(sizeof(float) * graph.num_nodes());
        sources  = (int *)  malloc(sizeof(int)   * graph.num_edges());
        targets  = (int *)  malloc(sizeof(int)   * graph.num_edges());
        fx       = (float *)malloc(sizeof(float) * graph.num_nodes());
        fy       = (float *)malloc(sizeof(float) * graph.num_nodes());
        fx_prev  = (float *)malloc(sizeof(float) * graph.num_nodes());
        fy_prev  = (float *)malloc(sizeof(float) * graph.num_nodes());

        for (nid_t n = 0; n <  graph.num_nodes(); ++n)
        {
            posx[n] = get_random(-width/2.0,  width/2.0);
            posy[n] = get_random(-height/2.0, height/2.0);
            mass[n] = graph.degree(n) + 1;
            fx[n] = 0.0;
            fy[n] = 0.0;
            fx_prev[n] = 0.0;
            fy_prev[n] = 0.0;
        }
        
        int cur_sources_idx = 0;
        int cur_targets_idx = 0;
        
        // Initialize the sources and targets arrays with edge-data.
        for (nid_t source_id = 0; source_id < graph.num_nodes(); ++source_id)
        {
            for (int eid = 0; eid < graph.degree(source_id); ++eid)
            {
                nid_t target_id = graph.nbr_id_for_node(source_id, eid);
                if (source_id > target_id) continue;
                sources[cur_sources_idx++] = source_id;
                targets[cur_targets_idx++] = target_id;
            }
        }
        
        
        // Allocate space on device.
        int max_threads_per_block;
        cudaCatchError(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0));

        nnodes = nbodies * 2;
        if (nnodes < max_threads_per_block*mp_count) nnodes = max_threads_per_block*mp_count;
        while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
        nnodes--;
        // nnodes+1 is now the number of nodes, and nnodes now points
        // to the last element in the massd, posxd, posyd, countd, startd and sortd arrays.
        // childd 4*nnodes is the index of the first child of the root node.
        
        // child stores pointer structure of the quadtree.
        cudaCatchError(cudaMalloc((void **)&childl,  sizeof(int)   * (nnodes+1) * 4));
        
        // the following properties, for each node in the quadtree (both internal and leaf)
        cudaCatchError(cudaMalloc((void **)&massl,   sizeof(float) * (nnodes+1)));
        cudaCatchError(cudaMalloc((void **)&posxl,   sizeof(float) * (nnodes+1)));
        cudaCatchError(cudaMalloc((void **)&posyl,   sizeof(float) * (nnodes+1)));
        // count contains the number of nested nodes for each node in quadtree
        cudaCatchError(cudaMalloc((void **)&countl,  sizeof(int)   * (nnodes+1)));
        // start contains ...
        cudaCatchError(cudaMalloc((void **)&startl,  sizeof(int)   * (nnodes+1)));
        cudaCatchError(cudaMalloc((void **)&sortl,   sizeof(int)   * (nnodes+1)));
        
        
        cudaCatchError(cudaMalloc((void **)&sourcesl,sizeof(int)   * (nedges)));
        cudaCatchError(cudaMalloc((void **)&targetsl,sizeof(int)   * (nedges)));
        cudaCatchError(cudaMalloc((void **)&fxl,     sizeof(float) * (nbodies)));
        cudaCatchError(cudaMalloc((void **)&fyl,     sizeof(float) * (nbodies)));
        cudaCatchError(cudaMalloc((void **)&fx_prevl,sizeof(float) * (nbodies)));
        cudaCatchError(cudaMalloc((void **)&fy_prevl,sizeof(float) * (nbodies)));
        
        
        // Used for reduction in BoundingBoxKernel
        cudaCatchError(cudaMalloc((void **)&maxxl,   sizeof(float) * mp_count * FACTOR1));
        cudaCatchError(cudaMalloc((void **)&maxyl,   sizeof(float) * mp_count * FACTOR1));
        cudaCatchError(cudaMalloc((void **)&minxl,   sizeof(float) * mp_count * FACTOR1));
        cudaCatchError(cudaMalloc((void **)&minyl,   sizeof(float) * mp_count * FACTOR1));
        
        // Used for reduction in SpeedKernel
        cudaCatchError(cudaMalloc((void **)&swgl,    sizeof(float) * mp_count * FACTOR6));
        cudaCatchError(cudaMalloc((void **)&etral,   sizeof(float) * mp_count * FACTOR6));
        
        // Copy host data to device.
        cudaCatchError(cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(sourcesl, sources, sizeof(int) * nedges, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(targetsl, targets, sizeof(int) * nedges, cudaMemcpyHostToDevice));

        // cpy fx, fy , fx_prevl, fy_prevl so they are all initialized to 0 in device memory.
        cudaCatchError(cudaMemcpy(fxl, fx,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fyl, fy,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fx_prevl, fx_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fy_prevl, fy_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
    }
    
    void CUDAFA2Layout::freeGPUMemory()
    {
        cudaFree(childl);
        
        cudaFree(massl);
        cudaFree(posxl);
        cudaFree(posyl);
        cudaFree(sourcesl);
        cudaFree(targetsl);
        cudaFree(countl);
        cudaFree(startl);
        cudaFree(sortl);
        
        cudaFree(fxl);
        cudaFree(fx_prevl);
        cudaFree(fyl);
        cudaFree(fy_prevl);
        
        cudaFree(maxxl);
        cudaFree(maxyl);
        cudaFree(minxl);
        cudaFree(minyl);
        
        cudaFree(swgl);
        cudaFree(etral);
    }
    
    CUDAFA2Layout::~CUDAFA2Layout()
    {
        free(mass);
        free(posx);
        free(posy);
        free(sources);
        free(targets);
        free(fx);
        free(fy);
        free(fx_prev);
        free(fy_prev);
        
        freeGPUMemory();
        
    }
    
    void CUDAFA2Layout::benchmark()
    {
        printf("Using %d MPs\n", mp_count);
        const int num_reps = 5;
        
        float times[11] = {0.0, };
        float time; // to temporarily hold time of a kernel
        
        const char *kernel_names[11] = {"Gravity", "Attractive", "BoundingBox", "Clear1", "TreeBuilding", "Clear2", "Summarization", "Sort", "Force", "Speed", "Displacement"};
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaDeviceSynchronize();
        auto starttime = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_reps; ++i)
        {
            cudaEventRecord(start, 0);
            GravityKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, k_g, strong_gravity, massl, posxl, posyl, fxl, fyl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[0] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            AttractiveForceKernel<<<mp_count * FACTOR6, THREADS6>>>(nedges, posxl, posyl, massl, fxl, fyl, sourcesl, targetsl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[1] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            BoundingBoxKernel<<<mp_count * FACTOR1, THREADS1>>>(nnodes, nbodies, startl, childl, massl, posxl, posyl, maxxl, maxyl, minxl, minyl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[2] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            ClearKernel1<<<1024, 1>>>(nnodes, nbodies, childl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[3] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            TreeBuildingKernel<<<mp_count * FACTOR2, THREADS2>>>(nnodes, nbodies, childl, posxl, posyl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[4] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            ClearKernel2<<<1024, 1>>>(nnodes, startl, massl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[5] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            SummarizationKernel<<<mp_count * FACTOR3, THREADS3>>>(nnodes, nbodies, countl, childl, massl, posxl, posyl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[6] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            SortKernel<<<mp_count * FACTOR4, THREADS4>>>(nnodes, nbodies, sortl, countl, startl, childl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[7] += time;
            cudaCatchError(cudaGetLastError());

            float epssq  = 0.05 * 0.05;            // Some sort of softening (eps, squared)
            float itolsq = 1.0f / (theta * theta); // Inverse tolerance, squared
            cudaEventRecord(start, 0);
            ForceCalculationKernel<<<mp_count * FACTOR5, THREADS5>>>(nnodes, nbodies, itolsq, epssq, sortl, childl, massl, posxl, posyl, fxl, fyl, k_r);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[8] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            SpeedKernel<<<mp_count * FACTOR1, THREADS1>>>(nbodies, fxl, fyl, fx_prevl, fy_prevl, massl, swgl, etral);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[9] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            DisplacementKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, posxl, posyl, fxl, fyl, fx_prevl, fy_prevl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[10] += time;
            cudaCatchError(cudaGetLastError());
        }
        
        cudaDeviceSynchronize(); // Not really neccesary given the preceding cudaEventSynchronize(stop).
        auto endtime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runningtime = (endtime - starttime);
        std::chrono::microseconds runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        
        printf("Benchmarking Results (averaging %d times):\n", num_reps);
        printf("\tkernel durations (us), resolution: ... us.:\n");
        for (int i = 0; i < 11; ++i)
        {
            printf("%s %.4f\n", kernel_names[i], 1000.0 * times[i] / (float)num_reps);
        }
        printf("\n");
        printf("Total %.2f\n", runningtime_us.count() / (float)num_reps);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void CUDAFA2Layout::doStep()
    {
        int err = 0;

        /******************************************************************************/
        /*** Compute Gravitational Force **********************************************/
        /******************************************************************************/
        GravityKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, k_g, strong_gravity, massl, posxl, posyl, fxl, fyl);
    
        /******************************************************************************/
        /*** Compute Attractive Force  ************************************************/
        /******************************************************************************/
        AttractiveForceKernel<<<mp_count * FACTOR6, THREADS6>>>(nedges, posxl, posyl, massl, fxl, fyl, sourcesl, targetsl);
        
        /******************************************************************************/
        /*** Compute bounding box *****************************************************/
        /******************************************************************************/
        BoundingBoxKernel<<<mp_count * FACTOR1, THREADS1>>>(nnodes, nbodies, startl, childl, massl, posxl, posyl, maxxl, maxyl, minxl, minyl);
        
        /******************************************************************************/
        /*** Build Barnes-Hut Tree ****************************************************/
        /******************************************************************************/
        // Set all child pointers of internal nodes (in childl) to null (-1)
        ClearKernel1<<<mp_count, 1024>>>(nnodes, nbodies, childl);
        
        // Build the tree
        TreeBuildingKernel<<<mp_count * FACTOR2, THREADS2>>>(nnodes, nbodies, childl, posxl, posyl);
        
        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpyFromSymbol(&err, errd, sizeof(int), 0, cudaMemcpyDeviceToHost));
        if (err != 0)
        {
            fprintf(stderr, "error: An error occurred in TreeBuildingKernel, errd == :%d\n", err);
         //   exit(EXIT_FAILURE);
        }
        
        //  Set all cell mass values to -1.0, set all startd to null (-1)
        ClearKernel2<<<mp_count, 1024>>>(nnodes, startl, massl);
        
        /******************************************************************************/
        /*** Compute Mass for all Cells ***********************************************/
        /******************************************************************************/
        SummarizationKernel<<<mp_count * FACTOR3, THREADS3>>>(nnodes, nbodies, countl, childl, massl, posxl, posyl);
        
        /******************************************************************************/
        /*** Sort the Tree ************************************************************/
        /******************************************************************************/
        SortKernel<<<mp_count * FACTOR4, THREADS4>>>(nnodes, nbodies, sortl, countl, startl, childl);
        
        /******************************************************************************/
        /*** Compute Repulsive Force **************************************************/
        /******************************************************************************/
        float epssq  = 0.05 * 0.05;            // Some sort of softening (eps, squared)
        float itolsq = 1.0f / (theta * theta); // Inverse tolerance, squared
        ForceCalculationKernel<<<mp_count * FACTOR5, THREADS5>>>(nnodes, nbodies, itolsq, epssq, sortl, childl, massl, posxl, posyl, fxl, fyl, k_r);

        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpyFromSymbol(&err, errd, sizeof(int), 0, cudaMemcpyDeviceToHost));
        if (err != 0)
        {
            fprintf(stderr, "error: An error occurred in ForceCalculationKernel, errd == %d\n", err);
            exit(EXIT_FAILURE);
        }


        /******************************************************************************/
        /*** Update Speeds ************************************************************/
        /******************************************************************************/
        SpeedKernel<<<mp_count * FACTOR1, THREADS1>>>(nbodies, fxl, fyl, fx_prevl, fy_prevl, massl, swgl, etral);
        
        /******************************************************************************/
        /*** Update Positions *********************************************************/
        /******************************************************************************/
        DisplacementKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, posxl, posyl, fxl, fyl, fx_prevl, fy_prevl);
        
        iteration++;
    }
    
    void CUDAFA2Layout::doSteps(int n)
    {
        for (int i = 0; i < n; ++i) doStep();
    }
    
    void CUDAFA2Layout::setScale(float s)
    {
        k_r = s;
    }
    
    void CUDAFA2Layout::setGravity(float g)
    {
        k_g = g;
    }
    
    void CUDAFA2Layout::retrieveLayoutFromGPU()
    {
        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
        cudaCatchError(cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
        cudaCatchError(cudaMemcpy(fx, fxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
        cudaCatchError(cudaMemcpy(fy, fyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
        cudaCatchError(cudaMemcpy(fx_prev, fx_prevl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
        cudaCatchError(cudaMemcpy(fy_prev, fy_prevl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
    }
    
    void CUDAFA2Layout::sendLayoutToGPU()
    {
        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fxl, fx,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fyl, fy,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fx_prevl, fx_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fy_prevl, fy_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
    }

    void CUDAFA2Layout::sendGraphToGPU()
    {
        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(sourcesl, sources, sizeof(int) * nedges, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(targetsl, targets, sizeof(int) * nedges, cudaMemcpyHostToDevice));
    }
    
        
    void CUDAFA2Layout::writeToPNG(const int width, const int height, const char *path)
    {
        cudaDeviceSynchronize(); // Wait for all kernels to complete.
        ComputeLayoutDimensions<<<mp_count * FACTOR1, THREADS1>>>(nbodies, posxl, posyl, maxxl, maxyl, minxl, minyl);
        
        // Retrieve data form GPU
        float minx_h, maxx_h, miny_h, maxy_h;
        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpyFromSymbol(&minx_h, minxdg, sizeof(float)));
        cudaCatchError(cudaMemcpyFromSymbol(&maxx_h, maxxdg, sizeof(float)));
        cudaCatchError(cudaMemcpyFromSymbol(&miny_h, minydg, sizeof(float)));
        cudaCatchError(cudaMemcpyFromSymbol(&maxy_h, maxydg, sizeof(float)));
        cudaCatchError(cudaMemcpy(posx, posxl,     sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
        cudaCatchError(cudaMemcpy(posy, posyl,     sizeof(float) * nbodies, cudaMemcpyDeviceToHost));
        float img_width = 5000;
        float img_height = 5000;
        
        const float xRange = maxx_h - minx_h;
        const float yRange = maxy_h - miny_h;
        const float xCenter = minx_h + xRange / 2.0;
        const float yCenter = miny_h + yRange / 2.0;
        const float minX = xCenter - xRange   / 2.0;
        const float minY = yCenter - yRange   / 2.0;
        const float xScale = img_width/xRange;
        const float yScale = img_height/yRange;
        
        // Here we need to do some guessing as to what the optimal
        // opacity of nodes and edges might be, given how many of them we need to draw.
        const float node_opacity = 1/(0.0001  * graph.num_nodes());
        const float edge_opacity = 1/(0.00001 * graph.num_edges());
        
        
        pngwriter layout_png(img_width, img_height, 0, path);
        layout_png.invert(); // set bg. to white.
        
        for (int n1 = 0; n1 < graph.num_nodes(); ++n1)
        {
            // Plot node,
            layout_png.filledcircle_blend((posx[n1] - minX)*xScale,
                                          (posy[n1] - minY)*yScale,
                                          3, node_opacity, 0, 0, 0);
            for (int e = 0; e < graph.degree(n1); ++e) {
                int n2 = graph.nbr_id_for_node(n1, e);
                if (n1 > n2) continue;
                // ... and edge.
                layout_png.line_blend((posx[n1] - minX)*xScale, (posy[n1] - minY)*yScale,
                                      (posx[n2] - minX)*xScale, (posy[n2] - minY)*yScale,
                                      edge_opacity, 0, 0, 0);
            }
        }
        // Write to file.
        layout_png.write_png();
    }
    
    void CUDAFA2Layout::writePositions(const char *path)
    {
        std::ofstream out_file(path);
        for (nid_t n = 0; n < graph.num_nodes(); ++n)
        {
            out_file << graph.offset_to_nid[n] << " " << posx[n] << " " << posy[n] << "\n";
        }
    }
}
