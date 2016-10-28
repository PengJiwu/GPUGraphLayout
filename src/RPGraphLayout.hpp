/*
 ==============================================================================
 
 RPGraphLayout.cpp
 
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

#ifndef RPGraphLayout_hpp
#define RPGraphLayout_hpp

#include "RPGraph.hpp"
#include "RPCommon.hpp"

namespace RPGraph
{
    class GraphLayout
    {
    private:
        Coordinate *coordinates;
        
    protected:
        float width, height;
        float minX(), minY(), maxX(), maxY();

    public:
        GraphLayout(RPGraph::CSRUGraph &graph, float width, float height);
        ~GraphLayout();
        
        CSRUGraph graph; // to lay-out
        
        // randomize the layout position of all nodes.
        void randomizePositions();
        
        float getX(nid_t node_id), getY(nid_t node_id);
        float getXRange(), getYRange();
        float getDistance(nid_t n1, nid_t n2);
        Real2DVector getDistanceVector(nid_t n1, nid_t n2);
        Real2DVector getNormalizedDistanceVector(nid_t n1, nid_t n2);
        Coordinate getCoordinate(nid_t node_id);
        Coordinate getCenter();


        void setX(nid_t node_id, float x_value), setY(nid_t node_id, float y_value);
        void moveNode(nid_t, Real2DVector v);
        void setCoordinates(nid_t node_id, Coordinate c);
        void writeToPNG(const int width, const int height, const char *path);
        void writePositions(const char *path);

    };
}

#endif /* RPGraphLayout_hpp */