#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "graph/graph_index/data.h"
#include "graph/subgraph_build_merge/subgraph_operation.cuh"

using namespace std;

int main(int argc,char** argv){

    //required variables from external input
    string base_path = argv[1];
    string subgraph_path = argv[2];
    string final_graph_path = argv[3];
    int num_of_candidates = atoi(argv[4]);
    int num_of_initial_neighbors = atoi(argv[5]);
    int num_of_points = atoi(argv[6]);
    // subgraph* sg;
    // sg = new subgraph(base_path, subgraph_path, final_graph_path);
    // sg->subgraphBuild(num_of_candidates, num_of_initial_neighbors, num_of_points);
    
    Data* points = new Data(base_path,num_of_points);

    GraphWrapper* graph;
    graph = new NavigableSmallWorldGraphWithFixedDegree(points);
	
    cout << "Construct proximity graph ..." << endl << endl;
    graph->Establishment(num_of_initial_neighbors, num_of_candidates);
    
    cout << "Save proximity graph ..." << endl << endl;
    string graph_path="/home/ErHa/GANNS_Res/bigann_base.bvecs";
    string graph_name = graph_path+"_"+argv[4]+"_"+argv[5]+"_"+to_string(num_of_points/1000000)+"M"+".nsw";
    graph->Dump(graph_name, num_of_initial_neighbors);
    cout << "Done" << endl;
    
    return 0;
}
