#include <algorithm>
#include <iostream>
#include <map>
#include <stack>
#include <sstream>
#include <climits>
#include <vector>
#include <set>
#include <signal.h>
#include <unistd.h>
#include <cstring>
#include <cmath>

volatile sig_atomic_t tle = 0;

#define LEVEL 0 // 1 - print all 0 -- submission level
#define DEBUG if(LEVEL) 
#define LOCAL 0
#define TEST if(LOCAL)

using namespace std;

class Edge { 
	
public:
	int to;
	int length; 
	
	Edge(){}
	~Edge(){}
	Edge(int t, int l){
		to = t; length = l;
	}
	bool operator < (const Edge& e){
		return length < e.length;
	}
};


vector< vector<Edge> > graph;
map<pair<int,int> , int> W;

void printAdjList(const vector< vector<Edge> > &graph){
	int i = 0;
	for (auto vec : graph){
		
		cout << i << ": ";
		for(auto e : vec){
			cout<< e.to << " ";
		}
		i++;
		cout << endl;
	}
}
void printTerminals(vector <int> & terminals){
	for(auto v : terminals){
		cout << v << "[shape=\"doublecircle\"]" << endl;
	}
}
void printEdgeList(const vector< vector<Edge> > &graph, bool withWeight=false, bool isViz= false){
	for(int i=0, endI = graph.size(); i < endI; i++){
		for(int j=0, endJ = graph[i].size(); j < endJ; j++){
			if(i < graph[i][j].to){
			//~ cout << i << " -- "<< e.to << ": " << e.length << endl;
				if(withWeight){
					cout << i << " "<< graph[i][j].to << " : " << graph[i][j].length <<  endl;
				}else if(isViz){
					cout << i << " -- "<< graph[i][j].to << "[label=" << graph[i][j].length << ",weight="<<  graph[i][j].length << ",color=red, penwidth=2]" <<  endl;
				}else {
					cout << i << " "<< graph[i][j].to <<  endl;
				}
			}
		}
	}
	
}

int getGraphWeight(const vector< vector<Edge> > &graph){
	int mstVal =0;
	
	for(int i=0, endI = graph.size(); i < endI; i++){
		for(int j=0, endJ = graph[i].size(); j < endJ; j++){
			if(i < graph[i][j].to)
				mstVal += graph[i][j].length;
		}
	}
	
	return mstVal;
}



void term(int signum)
{
	//~ cout << "VALUE " << minCost << endl;
	//~ auto nG = inducedSubgraph(graph, steinerTreeV);
	//~ auto mst = PrimsAlgo(nG, W, *(steinerTreeV.begin()));
	//~ printEdgeList(mst);
	
	exit(0);
} 

vector<int> dijkstra(
const vector< vector<Edge> > &graph, int source, 
	int target, vector<int>& min_distance) {
	vector<int> parent(graph.size() , -1);	
	min_distance[ source ] = 0;
	set< pair<int,int> > active_vertices;
	active_vertices.insert( {0,source} );
	
	while (!active_vertices.empty()) {
		int where = active_vertices.begin()->second;
		
		active_vertices.erase( active_vertices.begin() );
		for (auto ed : graph[where]) {
			auto newdist = min_distance[where] + ed.length;
			if (newdist < min_distance[ed.to]) {
				active_vertices.erase( { min_distance[ed.to], ed.to } );
				min_distance[ed.to] = newdist;
				parent[ed.to] = where;
				active_vertices.insert( { newdist, ed.to } );
			}
		}
	}
	return parent;
}
vector<int> BellmanFordMoore(
const vector< vector<Edge> > &graph, int source, 
	int target, vector<int>& minDist) {
	int N = graph.size();
	vector<int> parent(N, -1);	
	// all parent are -1
	// all minDist are INT_MAX
	
	minDist[source] =0;
	bool updated = true;
	for (int k =1; k < (N-2) && updated; k++){
		updated = false;
		cout << k <<" of " << N-2 <<endl;
		for(int u=1, endU = N; u < endU; u++){
			for(int j=0, endJ = graph[u].size(); j < endJ; j++){
				int w = graph[u][j].length; // edge weight of (u,v)
				int v = graph[u][j].to;
				int newDist =  minDist[u]+w;
				if(	newDist < minDist[v] ){ // the to perform relax!
					minDist[v] = newDist;
					parent[v] = u; 
					updated=true;
				}
			}
		}
	}
	return parent;
}
bool checkParent(const vector <int> &minDistD,  
				 const vector <int> &minDistB ){
	
	int N = minDistD.size();
	for(auto i=1; i !=N ; i++){
		//~ cout << parentD[i] << " == " << parentB[i]<< endl; 
		if(minDistD[i] != minDistB[i] )	
			return false;
	}
	return true;
}
int main(){
    struct sigaction action;
    memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = term;
    sigaction(SIGTERM, &action, NULL);
 
    ios_base::sync_with_stdio(false);

	vector<vector<int>> tdEdge;
	vector<vector<int>> tdBag;
	vector <int> terminals;
	set <int> terminalSet;
	string code, type, dummy;
	int N;
	 
	while( cin>> code >> type ){
	
		if(code == "SECTION" && type =="Graph"){
			long m, n;
			long u, v, w;
			cin >> dummy >> n;
			cin >> dummy >> m;
			N = n+1; 
			graph.resize(N); // coz graph has from index 0. where as challege its 1
			for(long i=0; i < m; i++){
				cin>> dummy >> u >> v >> w;
				graph[u].push_back(Edge(v,w));
				graph[v].push_back(Edge(u,w));
				W[make_pair(u,v)]=w;
				W[make_pair(v,u)]=w;
			}
			cin >> dummy >> ws;
		}
		else if(code == "SECTION" && type =="Terminals"){
			long t, u;
			cin >> dummy >> t;
			for(long i=0; i < t; i++){
				cin>> dummy >> u;
				terminals.push_back(u);
				terminalSet.insert(terminalSet.end(), u);
			}
			cin >> dummy >> ws;
		}
		else if(code == "SECTION" && type =="Tree"){
			
			cin >> dummy >> ws; // DECOMP
			cin >> dummy >> dummy; // s ,td
			long bags, bid, val ; 
			cin >> bags; // total bags
			tdEdge.resize(bags+1);
			tdBag.resize(bags+1);
			cin >> dummy >> dummy >> ws; // tw, n
			
			for(long i=0; i < bags; i++){
				string line;
				getline(cin, line); stringstream sstream(line);
				if(sstream >> dummy, dummy=="b"){
					sstream >> bid; // bag id
					//~ cout << bid << ": ";
					while(sstream >> val){
						//~ cout << val << " " ;
						tdBag[bid].push_back(val);
					}
					//~ cout << endl;
				}
			}
			long tu, tv;
			for(long i=1; i < bags; i++){ // b-1 edges is Td
				cin >>  tu >> tv;
				//~ cout<<  tu << " " << tv << endl;
				tdEdge[tu].push_back(tv);
				tdEdge[tv].push_back(tu);
			}
			cin >> dummy >> ws; // END
			cin >> dummy >> ws; // eof
			
		}
		else{
			cout << "Err in Input/Parsing!" << endl;
			exit(1);
		}
	
	}
	
	vector <int> minDistB(N,INT_MAX/2);
	auto parentB = BellmanFordMoore(graph, 1, -1, minDistB);
	
	vector <int> minDistD(N,INT_MAX);
	auto parentD = dijkstra(graph, 1, -1, minDistD);
	if(checkParent(minDistD, minDistB))
		cout << "MATCHED"<<endl;
	else
		cout << "NOT MATCHED"<<endl;	
	return 0;
}

