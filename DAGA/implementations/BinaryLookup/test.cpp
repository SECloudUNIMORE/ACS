#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iterator>
#include <list>
#include <random>
#include <sstream>
#include <set>
#include <utility>
#include <vector>

#define N 10

void help(const char *command_name){
    std::cerr << command_name << " <std::model>" << std::endl;
}

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

int compare_ngrams(const std::vector<uint8_t> &lhs, const std::vector<uint8_t> &rhs){
    for(uint8_t i = 0; i < N; i++){
        if(lhs[i] > rhs[i]) {
            return 1;
        }
        else if(lhs[i] < rhs[i]){
            return -1;
        }
    }
    return 0;
}

bool binary_search(const std::vector<uint8_t> *elem, const std::vector<std::vector<uint8_t>> mdl){
    unsigned long low = 0, high = mdl.size();
    int8_t cmp_val = -1;

    while(low <= high){
        unsigned long i = low + (high-low) / 2;
        cmp_val = compare_ngrams(mdl[i], *elem);

        if(cmp_val == 0){
            return 1;
        }
        else if(cmp_val < 0){
            low = i+1;
        }
        else{
            high = i-1;
        }
    }
    return 0;
}

int main(int argc, char *argv[]){
    if(argc!=2){
        std::cerr << "Invalid number of arguments" << std::endl;
        help(argv[0]);
        return(1);
    }

    const char *csv_model = argv[1];

    std::vector<std::vector<uint8_t>> mdl;
    std::string tmp;

    std::ifstream model(csv_model, std::ifstream::in);
    if(!model.is_open()){
        std::cout << "Error opening model" << std::endl;
        return -1;
    }

    while(model >> tmp){
        std::vector<uint8_t> vect_temp;
        std::stringstream ss(tmp.erase(tmp.size() -1));
        uint8_t tmp_val = 0;

        for (uint8_t i; ss >> i;) {
            if (i == ',') {
                vect_temp.push_back(tmp_val);
                tmp_val = 0;
            }
            else{
                if(tmp_val != 0)    tmp_val *= 10;
                tmp_val += i - '0';
            }
        }

        mdl.push_back(vect_temp);
    }

    std::cout << mdl.size() << std::endl;

    int64_t total_duration;
    auto start = std::chrono::high_resolution_clock::now();

    for(int k=0; k<1000; k++){
        std::vector<uint8_t> r = *select_randomly(mdl.begin(), mdl.end());
        int found = binary_search(&r, mdl);
        if(found != 1){
            std::cout<< "Item not found" << std::endl;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Total duration: " << duration.count() << "ms [1000]" << std::endl;

}

//int main() {
//    int found;
//    auto start = std::chrono::high_resolution_clock::now();
//
//    for(uint16_t i=0; i<TEST_SIZE; i++){
//        found = binary_search(test[i]);
//        if(found != 1){
//            std::cout << "Item not found" << std::endl;
//        }
//    }
//
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//    std::cout << "Total duration: " << duration.count() << "micros [1000]" << std::endl;
//}

