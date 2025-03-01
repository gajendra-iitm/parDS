#include<iostream>
// #include<bits/stdc++.h>
#include<vector>
#include<cstdlib>
#include<omp.h>
#include <sys/time.h>

#define msize 100000000	// map size
#define ninsert 100000000
#define ndelete 100000000
#define nsearch 100000000
#define batchsize 100000000
#define NRUNS 5

using namespace std;


//Structure of the Map Table
struct table{
    int key;
    int value;
    int fill;
    int delet;
};


//Structure of the Element 
struct element{
    int key;
    int value;
};

//Defining Lock for each index of Map Table 
omp_lock_t lock[msize];


struct table *mapTable;




class Map{

    private:
    

    public:
    Map(){

        //Creating Map Table Dynamically        
        mapTable = (struct table*)malloc(msize*sizeof(struct table));

        for(int i=0 ; i<msize ; i++){
            mapTable[i].key = -3;
            mapTable[i].value = -3;
            mapTable[i].fill = 0;
            mapTable[i].delet = 0;
        }

        
        //Initialization of locks using parallel threads
        #pragma omp for
        for(int i = 0; i < msize; ++i){
            omp_init_lock(&lock[i]);
        }
        

    }
    
    //Hash Function(Should not be linear)
    int doubleDashingFunction(int num, int i){

        int f1 = (num);
        // int f2 = (num%5);
        return (f1+(i*i))%msize;

    }

    void insertel(int insertSize, vector<vector<int>> &insertEl){

        //for Marking the index of Insertion Batch Element
        int *workDone;
        workDone = (int*)malloc(insertSize*sizeof(int));

        int *searchDone;
        searchDone = (int*)malloc(insertSize*sizeof(int));

        for(int i=0 ; i<insertSize ; i++){
            searchDone[i] = 0;
            workDone[i] = 0;
        }


        searchPairFun(insertSize, insertEl, searchDone);


        #pragma omp for
        for(int i=0 ; i<insertSize ; i++){

            //taking key and value from input
            int key, value, hashedVal;
            key = insertEl[i][0];
            value = insertEl[i][1];	

            //counter helps in probing
            int counter = 0;

            //Run the loop untill the insertion is not marked
            while(!workDone[i] && !searchDone[i]){

                // printf("tid = %d and key = %d\n", omp_get_thread_num(), key);
                hashedVal = doubleDashingFunction(key, counter);
                // printf(" hashedVal = %d \n", hashedVal);

                //Check the condition if the respective postion is occupied or not
                if(mapTable[hashedVal].fill==0 || (mapTable[hashedVal].fill==1 && mapTable[hashedVal].delet==1)){

                    //lock the respective index of Map Table where we have to insert
                    omp_set_lock(&lock[hashedVal]);

                    //Again checking so that other thread should not fill at same location
                    if(mapTable[hashedVal].fill==0 || (mapTable[hashedVal].fill==1 && mapTable[hashedVal].delet==1)){						

                        mapTable[hashedVal].key = key;
                        mapTable[hashedVal].value = value;

                        if(mapTable[hashedVal].fill==0)
                            mapTable[hashedVal].fill=1;
                        else
                            mapTable[hashedVal].delet=0;

                        workDone[i] = 1;

                        //Unset the lock
                        omp_unset_lock(&lock[hashedVal]);		

                    }
                    else{

                        counter++;
                        //After one round of traversal of whole Map Table, break out loop
                        if(counter>30){
                            //Unset the lock
                            omp_unset_lock(&lock[hashedVal]);	//I don't think it is required here					
                            break;	
                        }

                    }

                }else{

                    counter++;
                    //After one round of traversal of whole Map Table, break out loop
                    if(counter>30){
                            break;	
                    }

                }
                // printf("tid = %d and key = %d value = %d and work=%d and counter = %d\n", omp_get_thread_num(), key, value, workDone[i], counter);
            }

        }

    }



    void searchPairFun(int searchSize, vector<vector<int>> &searchEl, int *searchDone){


        // int searchSize = sizeof(searchEl)/sizeof(searchEl[0]);

        //Start of Directive to run all iteration in parallel 
        #pragma omp parallel for
        for (int i = 0; i < searchSize; ++i)
        {
            int key, value, hashedVal;

            key = searchEl[i][0];
            value = searchEl[i][1];


            int counter = 0;

            while(!searchDone[i]){

                hashedVal = doubleDashingFunction(key, counter);
                if(!mapTable[hashedVal].fill)
                    break;
                else{
                    if(mapTable[hashedVal].key == key && mapTable[hashedVal].value == value){
                        // cout<<"same";
                        searchDone[i]=1;
                    }else{

                        counter++;
                        //After one round of traversal of whole Map Table, break out loop
                        if(counter>30)
                            break;
                    }

                }
            }
            
        }
        //End of Directive to run all iteration in parallel 	

    }

    void searchPair(int searchSize, vector<vector<int>> &searchEl){


        // int searchSize = sizeof(searchEl)/sizeof(searchEl[0]);
        int *searchDone;
        searchDone = (int*)malloc(searchSize*sizeof(int));

        for(int i=0 ; i<searchSize ; i++){
            searchDone[i] = 0;
        }

        //Start of Directive to run all iteration in parallel 
        #pragma omp parallel for
        for (int i = 0; i < searchSize; ++i)
        {
            int key, value, hashedVal;

            key = searchEl[i][0];
            value = searchEl[i][1];


            int counter = 0;

            while(!searchDone[i]){

                hashedVal = doubleDashingFunction(key, counter);
                if(!mapTable[hashedVal].fill)
                    break;
                else{
                    if(mapTable[hashedVal].key == key && mapTable[hashedVal].value == value){
                        // cout<<"same";
                        searchDone[i]=1;
                    }else{

                        counter++;
                        //After one round of traversal of whole Map Table, break out loop
                        if(counter>30)
                            break;
                    }

                }
            }
            
        }
        //End of Directive to run all iteration in parallel

	/*
        for(int i=0 ; i<searchSize ; i++){
            if(searchDone[i]){
                cout<<searchEl[i][0]<<" "<<searchEl[i][1]<<endl;
            }
        }
	*/

    }

    void searchKey(int searchSize, vector<int> &searchEl){

        int *searchDone,*checkSearch;
        searchDone = (int*)malloc(searchSize*sizeof(int));
        checkSearch = (int*)malloc(msize*sizeof(int));

        for(int i=0 ; i<msize ; i++){
            checkSearch[i] = 0;
        }

        for(int i=0 ; i<searchSize ; i++){
            searchDone[i] = 0;
            checkSearch[i] = 0;
        }

        
        int ansCounter = 0;

        //Start of Directive to run all iteration in parallel 
        #pragma omp parallel for
        for(int i=0 ; i<searchSize ; i++){

            int key = searchEl[i];
            int hashedVal, counter=0;


            while(1){

                hashedVal = doubleDashingFunction(key, counter);

                if(mapTable[hashedVal].fill==0){
                    break;
                }
                else{

                    if(mapTable[hashedVal].key == key){

                        //printf("\nmatch key = %d\n", key);
                        checkSearch[hashedVal] = 1;

                        //Directive to put atomic condition on ansCounter variable
                        #pragma omp atomic
                        ansCounter++;
                        counter++;	
                        searchDone[i] = 1;
                    }
                    else{
                        counter++;
                        //After one round of traversal of whole Map Table, break out loop
                        if(counter>30)
                            break;
                    }
                }
            }
        }
        //End of Directive to run all iteration in parallel 
	/*
        cout<<endl;
        for(int i=0 ; i<msize ; i++){
            if(checkSearch[i])
                cout<<mapTable[i].key<<" "<<mapTable[i].value<<endl;
        }
	*/

    }

    

    void deletePair(int deletBatchSize, vector<vector<int>> &deletBatchEl){
        //Start of Directive to run all iteration in parallel 
        #pragma omp parallel for 
        for(int i=0 ; i<deletBatchSize ; i++){

            int key = deletBatchEl[i][0];
            int value = deletBatchEl[i][1];

            int counter = 0;

            int hashedVal;

            while(1){

                hashedVal = doubleDashingFunction(key, counter);
                
                if(mapTable[hashedVal].fill == 0)
                    break;

                if(mapTable[hashedVal].key == key && mapTable[hashedVal].value == value){
                    mapTable[hashedVal].key = -3;
                    mapTable[hashedVal].value = -3;
                    mapTable[hashedVal].delet = 1;
                    break;
                }

                counter++;
                //After one round of traversal of whole Map Table, break out loop
                if(counter>30)
                    break;

            }

        }
        //End of Directive to run all iteration in parallel 



    }


    void deleteKey(int deletBatchSize, vector<int> &deletBatchEl){

        //Start of Directive to run all iteration in parallel 
        #pragma omp parallel for 
        for(int i=0 ; i<deletBatchSize ; i++){

            int key = deletBatchEl[i];				
            int counter = 0;
            int hashedVal;

            while(1){

                hashedVal = doubleDashingFunction(key, counter);

                if(mapTable[hashedVal].fill == 0)
                    break;

                if(mapTable[hashedVal].key == key && mapTable[hashedVal].delet != 1){
                    mapTable[hashedVal].key = -3;
                    mapTable[hashedVal].value = -3;
                    mapTable[hashedVal].delet = 1;						
                }

                counter++;
                //After one round of traversal of whole Map Table, break out loop
                if(counter>30)
                    break;
            }
        }
        //End of Directive to run all iteration in parallel 


    }

    void show(){

        cout<<endl;
        for(int i=0 ; i<msize ; i++){
            if(mapTable[i].fill==1 && mapTable[i].key!=-3){
                cout<<"key = "<<mapTable[i].key<<" value = "<<mapTable[i].value<<endl;
            }
        }

    }
    

};
    double rtclock(){
        struct timezone Tzp;
        struct timeval Tp;
        int stat;
        stat = gettimeofday(&Tp, &Tzp);
        if (stat != 0) printf("Error return from gettimeofday: %d", stat);
        return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
    }

    void printtime(const char *str, double starttime, double endtime){
        printf("%s%3f seconds\n", str, endtime - starttime);
    }




 int main(){

   for (int ii = 0; ii < NRUNS; ++ii) {
     cout << "Run " << ii << "/" << (NRUNS - 1) << "-----" << endl;
     Map mp;

     // printf("\nInsert Again\n");

     //Insertion Batch Size
     int insertSize = ninsert;
     vector<vector<int>> insertEl(insertSize, vector<int>(2,0));
    
     // int insertEl[insertSize][2];

     // t1 = omp_get_wtime();

    


     /*Start of Dummy Input*/
     insertEl[0][0] = 59;
     insertEl[0][1] = 6;
     // workDone[0] = 0;
     // searchDone[0] = 0;

     insertEl[1][0] = 59;
     insertEl[1][1] = 8;
     // workDone[1] = 0;
     // searchDone[1] = 0;

     insertEl[2][0] = 59;
     insertEl[2][1] = 9;
     // workDone[2] = 0;
     // searchDone[2] = 0;

     for(int i=3 ; i<insertSize ; i++){
         insertEl[i][0] = i;
         insertEl[i][1] = i+7;
         // workDone[i] = 0;	//This is important
         // searchDone[i] = 0;
     }
     // /*End of Dummy Input*/

    
     // //cout<<"size = "<<sizeof(insertEl)/sizeof(insertEl[0]);
     // int sz = sizeof(insertEl)/sizeof(insertEl[0]);
     int sz=batchsize;

	double starttime, endtime;

	cout << "Number of insertions = " << sz << endl;
     starttime = rtclock();
     mp.insertel(sz, insertEl);
     endtime = rtclock();
     printtime("Insertion time: ", starttime, endtime);


     //mp.show();

     // cout<< "\nnew1\n";
     //mp.insertel(sz, insertEl);
     //mp.show();
     // cout<< "\nnew2\n";

     for(int i=0 ; i<insertSize ; i++){
         insertEl[i][0] = i;
         insertEl[i][1] = i+7;
     }

	cout << "Number of deletions = " << sz << endl;
     starttime = rtclock();
     mp.searchPair(sz, insertEl);
     endtime = rtclock();
     printtime("Search time: ", starttime, endtime);


     // mp.deletePairFun(insertSize, insertEl);

     //mp.show();

     // cout<<"new3\n";

     vector<int> deletBatchEl(sz, 3);


     // int insertSizee = 2;
     // int insertEle[insertSize];

     // insertEle[0] = 59;
     // insertEle[1] = 3;

     for(int i=0 ; i<sz ; i++){
         deletBatchEl[i] = i;
     }

 
	cout << "Number of searches = " << sz << endl;
     starttime = rtclock();
     mp.deleteKey(3, deletBatchEl);
     endtime = rtclock();
     printtime("Delete time: ", starttime, endtime);

     // mp.show();

     //mp.searchKey(3, deletBatchEl);
     cout << endl;
  }

     // cout<<"hellp";
     return 0;
 }
