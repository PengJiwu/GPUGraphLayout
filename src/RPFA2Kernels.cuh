/*
 ==============================================================================
 
 RPFA2Kernels.cuh
 
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

/// Some variables for FA2 related to `speed'
__device__ float total_swingingd = 0.0;
__device__ float total_effective_tractiond = 0.0;
__device__ float k_s_maxd = 10.0;
__device__ float global_speedd = 1.0;
__device__ float speed_efficiencyd = 1.0;
__device__ float jitter_toleranced = 1.0;


/*** The Kernel Definitions ***/

/******************************************************************************/
/*** gravity kernel ***********************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS6, FACTOR6)
void GravityKernel(int nbodiesd, const float k_g, const bool strong_gravity,
                   volatile float * __restrict massd,
                   volatile float * __restrict posxd, volatile float * __restrict posyd,
                   volatile float * __restrict fxd, volatile float * __restrict fyd)
{
    register int i, inc;
    
    // iterate over all bodies assigned to thread
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc)
    {
        const float px = posxd[i];
        const float py = posyd[i];
        
        // `f_g' is the magnitude of gravitational force
        float f_g;
        if(strong_gravity)
        {
            f_g = k_g * massd[i];
        }
        else // weak gravity
        {
            if (px != 0.0 || py != 0.0)
            {
                f_g = k_g * massd[i] * rsqrtf(px*px + py*py);
            }
            
            else
            {
                f_g = 0.0;
            }
        }
        
        fxd[i] += (-px * f_g);
        fyd[i] += (-py * f_g);
    }
}

__global__
__launch_bounds__(THREADS6, FACTOR6)
void AttractiveForceKernel(int nedgesd,
                           volatile float * __restrict posxd, volatile float * __restrict posyd,
                           volatile float * __restrict massd,
                           volatile float * __restrict fxd, volatile float * __restrict fyd,
                           volatile int * __restrict sourcesd, volatile int * __restrict targetsd)
{
    register int i, inc, source, target;
    // iterate over all edges assigned to thread
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nedgesd; i += inc)
    {
        source = sourcesd[i];
        target = targetsd[i];
        
        // dx and dy are distance to between the neighbors.
        const float dx = posxd[target]-posxd[source];
        const float dy = posyd[target]-posyd[source];
        
        // Force just depends linearly on distance.
        const float fsx = dx;
        const float fsy = dy;
        
        const float ftx = -dx;
        const float fty = -dy;
        
        
        // these memory accesses aren't coalesced...
        atomicAdd((float*)fxd+source, fsx);
        atomicAdd((float*)fyd+source, fsy);
        
        atomicAdd((float*)fxd+target, ftx);
        atomicAdd((float*)fyd+target, fty);
    }
}


/******************************************************************************/
/*** update speeds ************************************************************/
/******************************************************************************/
__global__
__launch_bounds__(THREADS1, FACTOR1)
void SpeedKernel(int nbodiesd,
                 volatile float * __restrict fxd , volatile float * __restrict fyd,
                 volatile float * __restrict fx_prevd , volatile float * __restrict fy_prevd,
                 volatile float * __restrict massd, volatile float * __restrict swgd, volatile float * __restrict etrad)
{
    register int i, j, k, inc;
    register float swg_thread, swg_body, etra_thread, etra_body, dx, dy, mass;
    // setra: effective_traction (in shared mem.)
    // sswg: swing per node (in shared mem.)
    __shared__ volatile float sswg[THREADS1], setra[THREADS1];
    
    // initialize with valid data (in case #bodies < #threads)
    swg_thread  = 0;
    etra_thread = 0;
    
    // scan all bodies
    i = threadIdx.x;
    inc = THREADS1 * gridDim.x;
    
    for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc)
    {
        mass = massd[j];
        
        dx = fxd[j] - fx_prevd[j];
        dy = fyd[j] - fy_prevd[j];
        swg_body = sqrtf(dx*dx + dy*dy);
        swg_thread += mass * swg_body;
        
        dx = fxd[j] + fx_prevd[j];
        dy = fyd[j] + fy_prevd[j];
        etra_body = sqrtf(dx*dx + dy*dy) / 2.0;
        etra_thread += mass * etra_body;
    }
    
    // reduction in shared memory
    sswg[i]  = swg_thread;
    setra[i] = etra_thread;
    
    for (j = THREADS1 / 2; j > 0; j /= 2)
    {
        __syncthreads();
        if (i < j)
        {
            k = i + j;
            sswg[i]  = swg_thread  = sswg[i]  + sswg[k];
            setra[i] = etra_thread = setra[i] + setra[k];
        }
    }
    
    // swg_thread and etra_thread are now the total swinging
    // and the total effective traction (accross all threads)
    
    // write block result to global memory
    if (i == 0)
    {
        k = blockIdx.x;
        swgd[k]  = swg_thread;
        etrad[k] = etra_thread;
        __threadfence();
        
        inc = gridDim.x - 1;
        if (inc == atomicInc(&blkcntd, inc))
        {
            swg_thread = 0;
            etra_thread = 0;
            
            for (j = 0; j <= inc; j++)
            {
                swg_thread  += swgd[j];
                etra_thread += etrad[j];
            }
            // we need to do some calculations to derive
            // from this the new global speed
            float estimated_optimal_jitter_tollerance = 0.05 * sqrtf(nbodiesd);
            float minJT = sqrtf(estimated_optimal_jitter_tollerance);
            float jt = jitter_toleranced * fmaxf(minJT,
                                                 fminf(k_s_maxd, estimated_optimal_jitter_tollerance * etra_thread / powf(nbodiesd, 2.0)
                                                       ));
            float min_speed_efficiency = 0.05;
            
            // `Protect against erratic behavior'
            if (swg_thread / etra_thread > 2.0)
            {
                if (speed_efficiencyd > min_speed_efficiency) speed_efficiencyd *= 0.5;
                jt = fmaxf(jt, jitter_toleranced);
            }
            
            // `Speed efficiency is how the speed really corrosponds to the swinging vs. convergence tradeoff.'
            // `We adjust it slowly and carefully'
            float targetSpeed = jt * speed_efficiencyd * etra_thread / swg_thread;
            
            if (swg_thread > jt * etra_thread)
            {
                if (speed_efficiencyd > min_speed_efficiency)
                {
                    speed_efficiencyd *= 0.7;
                }
            }
            else if (global_speedd < 1000)
            {
                speed_efficiencyd *= 1.3;
            }
            
            // `But the speed shouldn't rise much too quickly, ... would make convergence drop dramatically'.
            double max_rise = 0.5;
            global_speedd += fminf(targetSpeed - global_speedd, max_rise * global_speedd);
        }
    }
}

/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS6, FACTOR6)
void DisplacementKernel(int nbodiesd,
                       volatile float * __restrict posxd, volatile float * __restrict posyd,
                       volatile float * __restrict fxd, volatile float * __restrict fyd,
                       volatile float * __restrict fx_prevd, volatile float * __restrict fy_prevd)
{
    register int i, inc;
    register float factor, swg, dx, dy, fx, fy;
    register float global_speed = global_speedd;
    // iterate over all bodies assigned to thread
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc)
    {
        fx = fxd[i];
        fy = fyd[i];
        dx = fx - fx_prevd[i];
        dy = fy - fy_prevd[i];
        swg = sqrtf(dx*dx + dy*dy);
        factor = global_speed / (1.0 + sqrtf(global_speed * swg));
        
        posxd[i] += fx * factor;
        posyd[i] += fy * factor;
        fx_prevd[i] = fx;
        fy_prevd[i] = fy;
        fxd[i] = 0.0;
        fyd[i] = 0.0;
    }
}