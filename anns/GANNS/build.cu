#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "data.h"
#include "graph_index/navigable_small_world.h"

using namespace std;

int main(int argc,char** argv){

    //required variables from external input
    string base_path = argv[1];
    int num_of_candidates = atoi(argv[2]);
    int num_of_initial_neighbors = atoi(argv[3]);
    int num_of_points = atoi(argv[4]);

    cout << "Load data points..." << endl << endl;
    Data* points = new Data(base_path,num_of_points);

    GraphWrapper* graph;
    graph = new NavigableSmallWorldGraphWithFixedDegree(points);
    
	
    cout << "Construct proximity graph ..." << endl << endl;
    graph->Establishment(num_of_initial_neighbors, num_of_candidates);
    
    cout << "Save proximity graph ..." << endl << endl;
    string graph_path="/home/ErHa/GANNS_Res/bigann_base.bvecs";
    string graph_name = graph_path+"_"+argv[3]+"_"+argv[4]+"_"+to_string(num_of_points/1000000)+"M"+".nsw";
    graph->Dump(graph_name);
    cout << "Done" << endl;
    
    return 0;
}
