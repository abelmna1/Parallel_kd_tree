#include <random>
#include <cstdlib>
#include <cstdio>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <sys/mman.h>
#include <linux/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>
#include <random>
#include <vector>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <ratio>
#include <array>
#include <queue>
#include <math.h>
#include <thread>
#include <string>
#include <atomic>

std::atomic<int64_t> REMAINING_CORES(0);

struct input_file {
    std::string file_type;
    uint64_t file_id;
    uint64_t num_points;
    uint64_t num_dimensions;
    uint64_t num_neighbors;
    std::size_t data_size;
    float * data;
    input_file() : file_type(""), file_id(0), num_points(0), num_dimensions(0), num_neighbors(0), data_size(0), data(NULL) {}
    
    void print_file_data(){
        std::cout << "file type: " << file_type << std::endl;  
        std::cout << "file id: " << file_id << std::endl;
        std::cout << "num points: " << num_points << std::endl;
        std::cout << "num dimensions: " << num_dimensions << std::endl;
        if(file_type == "QUERY") std::cout << "num neighbors: " << num_neighbors << std::endl;
        std::size_t count = 0;
        for(uint64_t i = 0; i < num_points; ++i){
            std::cout << "point " << i << ":";
            for(uint64_t j = 0; j < num_dimensions; ++j){
                std::cout << " " << data[num_dimensions*i+j];
                ++count;
            }
            std::cout << std::endl;
        }
        std::cout << count << "==" << data_size << std::endl;
    }
};

class Node {
    public:
        float value;
        uint64_t current_dimension;
        char leaf;
        uint64_t start, end;       
        Node *left_child, *right_child;     
        //int64_t start, end;       
        Node() : value(0), current_dimension(0), leaf(0), start(-1), end(-1), left_child(NULL), right_child(NULL) {}
        Node(float v, float d) : value(v), current_dimension(d), leaf(0), start(-1), end(-1), left_child(NULL), right_child(NULL) {}
};

struct queue_elem{
    float * point;
    float dist_to_query;
    queue_elem() : point(NULL), dist_to_query(10000000) {}
    queue_elem(float *f, float dist) : point(f), dist_to_query(dist) {}
};

struct Comp{
    bool operator()(const queue_elem& a, const queue_elem& b){return a.dist_to_query < b.dist_to_query;}
};

typedef std::priority_queue<queue_elem, std::vector<queue_elem>, Comp> my_queue;

void print_queries(const input_file& query_file, const my_queue& queue){
    //std::cout << "queue size: " << query_file.num_neighbors << "=" << queue.size() << std::endl;
    if(queue.size() == 0){ std::cout << "EMPTY" << std::endl; return; }
    my_queue q = queue; queue_elem current_elem;    

    for(uint64_t i = 0; i < queue.size(); ++i){
        std::cout << " (";
        current_elem = q.top(); q.pop();
        for(uint64_t k = 0; k < query_file.num_dimensions; ++k){
            std::cout << " " << current_elem.point[k];
        }
        std::cout << ")\n";
    }
}

void print_tree(Node* root, Node* parent, const input_file& file, const std::vector<float*>& vect, int tab){
    if(!root) return;
    //std::cout << "tab: " << tab << std::endl;
    std::cout << std::setw(tab) << " Node value: " << root->value << ", dim: " << root->current_dimension;
    if(parent) std::cout << ", parent: " << parent->value;
    std::cout << std::endl;
    if(root->leaf){
        std::cout << std::setw(tab);
        //std::cout << std::setw(tab) << "remaining points: " << std::endl;
        for(uint64_t i = root->start; i < root->end; ++i){
            std::cout <<  " (";
            for(uint64_t j = 0 ; j < file.num_dimensions; ++j){
                std::cout << " " << vect[i][j];
            }
            std::cout << ")";
        }
        std::cout << std::endl;
    }   
    else{
        std::cout << std::setw(tab) << "node " << root->value << " LEFT KIDS\n"; 
        print_tree(root->left_child, root, file, vect, tab+4);
        std::cout << std::setw(tab) << "node " << root->value << "  RIGHT KIDS\n";
        print_tree(root->right_child, root,  file, vect, tab+4);
    }
}

float find_median(const std::vector<float*>& vect, uint64_t start, uint64_t end, uint64_t current_dim){
    std::default_random_engine generator;
    std::uniform_int_distribution<> distribution(start, end-1);
    uint64_t rand_point; float temp;
    float sample[10000];
    for(uint64_t i = 0; i < 10000; ++i){ 
        rand_point = distribution(generator);
        temp = *(vect[rand_point] + current_dim);
        sample[i] = temp;
    }
    std::nth_element(sample, sample+5000, sample+10000);
    return sample[5000];
}

void construct_tree(Node ** root, const input_file& file, std::vector<float*>& vect, uint64_t start, uint64_t end, uint64_t current_depth){
    uint64_t num_points = end-start;
    uint64_t current_dim = current_depth % file.num_dimensions;
    if(num_points <= 10){ 
        Node * temp = NULL;
        temp = new Node();
        temp->current_dimension = current_dim;
        temp->leaf = 1;
        temp->start = start; temp->end = end;
        *root = temp;
        return;
    }
    else{
        Node * temp = NULL;
        float median;
        if(num_points < 10000){ 
            uint64_t median_index = start + (num_points/2);
            std::nth_element(vect.begin()+start, (vect.begin()+start)+(num_points/2), vect.begin() + end,
                [=](const float* e1, const float* e2){ return e1[current_dim] < e2[current_dim]; });
    
            median = *(vect[median_index] + current_dim);
            temp = new Node(median, current_dim);
            *root = temp;
            if((REMAINING_CORES--) > 0){
                std::thread left_thread(construct_tree, &(temp->left_child), std::ref(file), std::ref(vect), start, median_index+1, current_depth+1);
                construct_tree(&(temp->right_child), file, vect, median_index+1, end, current_depth+1);    
                left_thread.join(); ++REMAINING_CORES;
           }
            else{ 
                ++REMAINING_CORES;
                construct_tree(&(temp->left_child), file, vect, start, median_index+1, current_depth+1);
                construct_tree(&(temp->right_child), file, vect, median_index+1, end, current_depth+1);    
            }
       }
       else{
            median = find_median(vect, start, end, current_dim);
            auto iter = std::partition(vect.begin()+start, vect.begin()+end, [&](const float *a) {return a[current_dim] <= median;});
            uint64_t right_vector_start = iter - vect.begin();
            temp = new Node(median, current_dim);
            *root = temp;
            if((REMAINING_CORES--) > 0){
                std::thread left_thread(construct_tree, &(temp->left_child), std::ref(file), std::ref(vect), start, right_vector_start, current_depth+1);
                construct_tree(&(temp->right_child), file, vect, right_vector_start, end, current_depth+1);
                left_thread.join(); ++REMAINING_CORES;
            }
            else{
                ++REMAINING_CORES;
                construct_tree(&(temp->left_child), file, vect, start, right_vector_start, current_depth+1);
                construct_tree(&(temp->right_child), file, vect, right_vector_start, end, current_depth+1);
            }
        }
        return;
    }
}

void query_search(const input_file& query_file, const uint64_t query_index, const std::vector<float*>& points, my_queue& queue, char& leaf_found, const Node * node){
    if(!node) return;
    float euclid_dist = 0, orthog_dist_node = 0, orthog_dist_max = 0, power_sum = 0, current_diff = 0;
    float * current_query = query_file.data + (query_index*query_file.num_dimensions); char more_neighbors_needed = 0;


    if(!node->leaf){
        //if queue is still not full, further traversal is needed
        if(queue.size() < query_file.num_neighbors){ 
            more_neighbors_needed = 1;     
            if(leaf_found == 1) leaf_found = 0;
        }
        if(!more_neighbors_needed){
            orthog_dist_node = fabs(current_query[node->current_dimension] - node->value);
            orthog_dist_max = (queue.top()).dist_to_query;
        }
        
        //check if |Q| < K or dist from max queue is greater than current dist
        if(more_neighbors_needed || orthog_dist_node < orthog_dist_max || !leaf_found){
            //query at k_dim < node->k_dim
            if(current_query[node->current_dimension] <= node->value){
                //std::cout << "TRAVERSE LEFT, THEN RIGHT\n";
                query_search(query_file, query_index, points, queue, leaf_found, node->left_child);
                if(queue.size() < query_file.num_neighbors || ( (queue.top()).dist_to_query > orthog_dist_node)){
                    leaf_found = 0;
                    query_search(query_file, query_index, points, queue, leaf_found, node->right_child);
                }
            }
            else{
                //std::cout << "TRAVERSE RIGHT, THEN LEFT\n";
                query_search(query_file, query_index, points, queue, leaf_found, node->right_child);
                 if(queue.size() < query_file.num_neighbors || ( (queue.top()).dist_to_query > orthog_dist_node)){
                    leaf_found = 0;
                    query_search(query_file, query_index, points, queue, leaf_found, node->left_child);
                 }
            }
        }
        else return;
    }
    else{
        //traverse leaf node & add points to queue
        for(uint64_t i = node->start; i < node->end; ++i){
            power_sum = 0; current_diff = 0; euclid_dist = 0;


            for(uint64_t j = 0; j < query_file.num_dimensions; ++j){
                current_diff = current_query[j] - points[i][j];
                power_sum += powf(current_diff, 2);
            }
            euclid_dist = sqrtf(power_sum);
            queue_elem new_elem(points[i], euclid_dist);            
            if(queue.size() == 0){ queue.push(new_elem); continue; }
            else{
                if(queue.size() < query_file.num_neighbors) queue.push(new_elem);
                else if(queue.size() == query_file.num_neighbors){
                    if(euclid_dist < (queue.top()).dist_to_query){
                        queue.pop(); queue.push(new_elem);
                    }        
                }        
            }   
        }
        leaf_found = 1;
    }
}

void find_queries(const input_file& query_file, const std::vector<float*>& points, float* results, Node* root, int64_t core, uint64_t queries_per_core,
    int64_t rem_queries) {
 
    uint64_t actual_remaining = (rem_queries) ? rem_queries : queries_per_core;
    uint64_t query_file_start = core*queries_per_core;  //*query_file.num_dimensions;
    uint64_t result_file_start = (core*queries_per_core*query_file.num_neighbors*query_file.num_dimensions);
    float * temp = results+result_file_start;

    for(uint64_t i = 0; i < actual_remaining; ++i, ++query_file_start){
        my_queue queue; char leaf_found = 0;
        query_search(query_file, query_file_start, points, queue, leaf_found, root);
        assert(queue.size() == query_file.num_neighbors);  //TODO assertion might not be needed

        for(uint64_t k = 0; k < query_file.num_neighbors; ++k){
            for(uint64_t m = 0; m < query_file.num_dimensions; ++m){      
                temp[m] = (queue.top()).point[m];
            }
            queue.pop();
            temp += query_file.num_dimensions;
        }
    }
}

input_file parse_file(const std::string &file) {
    int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) {
        int en = errno;
        std::cerr << "Couldn't open " << file << ": " << strerror(en) << "." << std::endl;
        exit(2);
    }
    struct stat sb;
    int rv = fstat(fd, &sb); assert(rv == 0);
    void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (vp == MAP_FAILED) {
        int en = errno;
        fprintf(stderr, "mmap() failed: %s\n", strerror(en));
        exit(3);
    }
    char *data = (char *) vp;

    // Tell the kernel that it should evict the pages as soon as possible.
    rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);
    rv = close(fd); assert(rv == 0);
    assert(sb.st_size % sizeof(float) == 0);
    std::size_t num_floats = sb.st_size / sizeof(float);

    int n = strnlen(data, 8);
    std::string file_type((char*)data, n); data+=8;
    input_file result_file;
    result_file.file_type = file_type;  

    result_file.file_id = *(uint64_t*)data; data+=sizeof(uint64_t);
    result_file.num_points = *(uint64_t*)data; data+=sizeof(uint64_t);
    result_file.num_dimensions = *(uint64_t*)data; data+=sizeof(uint64_t);

    if (result_file.file_type == "TRAINING") {
        result_file.data_size = num_floats-8; //32 init header bytes == 8 floats
        result_file.data = (float*) data;
        return result_file;
    }
    else if(result_file.file_type == "QUERY"){
        result_file.num_neighbors = *(uint64_t*)data; data+=sizeof(uint64_t);
        result_file.data_size = num_floats-10; //40 init header bytes == 10 floats
        result_file.data = (float*) data;
        return result_file;
    }
    else{
        std::cerr << file << " has unknown file type" << std::endl;
        exit(2);
    }   
}

float * construct_file(input_file& training_file, input_file& query_file, const std::string& file_name){
    std::ofstream out(file_name.c_str(), std::ios::binary);
    if (!out) { std::cerr << "couldn't open " << file_name << std::endl; exit(1); }
    char file_type[8] = {0}; strncpy(file_type, "RESULT", 6);
    uint64_t results_file_id;
    out.write(file_type, 8);
    out.write(reinterpret_cast<char*>(&(training_file.file_id)),8);
    out.write(reinterpret_cast<char*>(&(query_file.file_id)),8);
    std::ifstream urandom("/dev/urandom", std::ios::in| std::ios::binary);
    if(!urandom){ std::cout << "read from /dev/urandom failed\n"; exit(1); }
    if(urandom){
       urandom.read(reinterpret_cast<char*>(&results_file_id), 8);
       if(!urandom){ std::cout << "read from /dev/urandom failed\n"; exit(1);}
        out.write(reinterpret_cast<char*>(&results_file_id), 8);
    }
    out.write(reinterpret_cast<char*>(&(query_file.num_points)),8);
    out.write(reinterpret_cast<char*>(&(query_file.num_dimensions)),8);
    out.write(reinterpret_cast<char*>(&(query_file.num_neighbors)),8);
    out.close(); 
        
    int fd = open(file_name.c_str(), O_RDWR);
    if(fd < 0){ std::cerr << "couldn't open " << file_name << std::endl; exit(1); }


    uint64_t total_size = (query_file.num_points*query_file.num_neighbors*query_file.num_dimensions*sizeof(float))+56;
    if(ftruncate(fd, total_size) < 0){ std::cout << "error in truncating file\n"; exit(1); }
    
    char * rv = (char*) mmap(NULL, total_size, PROT_WRITE, MAP_SHARED|MAP_POPULATE, fd, 0);
    if(rv == MAP_FAILED){
        int en = errno;  
        fprintf(stderr, "error in mmap: %s\n", strerror(en));
        exit(1);
    }
    close(fd);    
    rv += (sizeof(uint64_t)*7);
    float * result_data = (float*) rv;
    return result_data;
}
 

int main(int argc, char **argv) {
    if(argc != 5){
        std::fprintf(stderr, "Usage: ./k-nn <n_cores> <training_file> <query_file> <result_file>\n");        
        exit(1);
    }

    int64_t num_cores = atoi(argv[1]);
    if(num_cores < 0){ std::cerr << "0 or more cores required\n"; exit(1); }
    if(num_cores == 0) num_cores = 1;
    REMAINING_CORES = (num_cores-1);

    input_file training_file = parse_file(argv[2]);
    input_file query_file = parse_file(argv[3]);
    if(query_file.num_dimensions != training_file.num_dimensions){ 
        std::cerr << "query and training files have different dimensions\n"; exit(1);
    }

    float * results = construct_file(training_file, query_file, argv[4]);

    std::vector<float*> training_points; training_points.reserve(training_file.num_points);
    float * temp = training_file.data;
    for(uint64_t i = 0; i < training_file.num_points; ++i, temp+=training_file.num_dimensions){
        training_points.push_back(temp);
    }  
    
    std::chrono::high_resolution_clock::time_point t1, t2, t3, t4;
	std::chrono::duration<double> insertion_time, query_time;
  
    Node * root = NULL;
    t1 = std::chrono::high_resolution_clock::now();
    construct_tree(&root, training_file, training_points, 0, training_file.num_points, 0);
    t2 = std::chrono::high_resolution_clock::now();
    insertion_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1);
    std::cout << "insertion time: " << insertion_time.count() << std::endl;

    if(query_file.num_points > 0){
        if(num_cores == 1){
            t3 = std::chrono::high_resolution_clock::now();
            find_queries(query_file, training_points, results, root, 0, query_file.num_points, 0); 
            t4 = std::chrono::high_resolution_clock::now();
        }
        else if((uint64_t) num_cores > query_file.num_points){
            std::vector<std::thread> cores; cores.reserve(query_file.num_points-1);
            t3 = std::chrono::high_resolution_clock::now();
            for(uint64_t i = 0; i < query_file.num_points-1; ++i){
                cores.emplace_back(std::thread(find_queries, std::ref(query_file), std::ref(training_points), results, root, i, 1, 0));
            }
            find_queries(query_file, training_points, results, root, query_file.num_points-1, 1, 0); 
            for(std::size_t i = 0; i < cores.size(); ++i) cores[i].join();
            t4 = std::chrono::high_resolution_clock::now();
        }
        else{
            std::vector<std::thread> cores; cores.reserve(num_cores-1); 
            uint64_t queries_per_core = query_file.num_points/num_cores;
            t3 = std::chrono::high_resolution_clock::now();
            for(int64_t i = 0; i < num_cores-1; ++i){
                cores.emplace_back(std::thread(find_queries, std::ref(query_file), std::ref(training_points), results, root, i, queries_per_core, 0));
            }
            uint64_t num_rem_queries = query_file.num_points - ((num_cores-1)*queries_per_core);
            find_queries(query_file, training_points, results, root, num_cores-1, queries_per_core, num_rem_queries);
            for(std::size_t i = 0; i < cores.size(); ++i) cores[i].join();
            t4 = std::chrono::high_resolution_clock::now();
        }
        query_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t4-t3);
        std::cout << "querying time: " << query_time.count() << std::endl; 
    }
        
   /* float * temp2 = results;
    for(uint64_t i = 0; i < query_file.num_points*query_file.num_neighbors*query_file.num_dimensions; ++i){
        if(i%2 == 0) std::cout << "(";
        std::cout << " " << temp2[i];
        if((i+1)%2 == 0) std::cout << ")\n";
    }*/
    

    struct rusage ru;
    int rv = getrusage(RUSAGE_SELF, &ru); assert(rv == 0);
    auto cv = [](const timeval &tv) {
        return double(tv.tv_sec) + double(tv.tv_usec)/1000000;
    };

    std::cerr << "Resource Usage:\n";
    std::cerr << "    User CPU Time: " << cv(ru.ru_utime) << '\n';
    std::cerr << "    Sys CPU Time: " << cv(ru.ru_stime) << '\n'; 
    std::cerr << "    Max Resident: " << ru.ru_maxrss << '\n';
    std::cerr << "    Page Faults: " << ru.ru_majflt << '\n';

    return 0;
}
