/*
 ==============================================================================
 
 RPCUDAForceAtlas2.hpp
 
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

#ifndef RPCUDAForceAtlas2_hpp
#define RPCUDAForceAtlas2_hpp
#include "RPGraph.hpp"
#include "RPGraphLayout.hpp"

namespace RPGraph
{
    class CUDAFA2Layout
    {
    private:
        CSRUGraph &graph;
        float width, height;
        
        /// CUDA Specific stuff.
        // Host storage.
        float *mass, *posx, *posy;
        float *fx, *fy, *fx_prev, *fy_prev;
        
        // Quick way to represent a graph on the GPU
        int *sources, *targets;
        
        // Host pointers to device memory (all suffixed with 'l').
        int   *errl,  *sortl, *childl, *countl, *startl;
        int   *sourcesl, *targetsl;
        float *massl, *posxl, *posyl;
        float *minxl, *minyl, *maxxl, *maxyl; // Used in reduction.
        float *fxl, *fyl, *fx_prevl, *fy_prevl;
        float *swgl, *etral;
        
        int mp_count; // Number of multiprocessors on GPU.
        int nnodes;
        int nbodies;
        int nedges;
        
        /// General FA2 stuf..
        int iteration;
        
        // Scalars for repulsive and gravitational force.
        float k_r, k_g;
        float delta; // edgeweight influence.
        float global_speed;
        
        // Parameters used in adaptive temperature
        float speed_efficiency, jitter_tolerance;
        float k_s, k_s_max; // magic constants related to swinging.
        float theta;        // an accuracy parameter used for BarnesHut.
    
    private:
        void GPUBindGraph(CSRUGraph graph);
        void sendGraphToGPU();
        void sendLayoutToGPU();
        void retrieveLayoutFromGPU();
        void freeGPUMemory();
        
    public:
        bool prevent_overlap, dissuade_hubs, use_barneshut, use_linlog, strong_gravity;
        
        CUDAFA2Layout(CSRUGraph &graph, float width, float height);
        ~CUDAFA2Layout();
        void doStep();
        void doSteps(int n);
        void setScale(float s);
        void setGravity(float s);
        void initGPU(CSRUGraph graph);
        
        
        void benchmark();
        void writeToPNG(const int width, const int height, const char *path);
        void writePositions(const char *path);
    };
};


#endif /* RPCUDAForceAtlas2_hpp */
