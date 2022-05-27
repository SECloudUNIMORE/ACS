#include <chrono>
#include <iostream>
#include <fstream>
#include <iterator>
#include <list>
#include <sstream>
#include <set>
#include <vector>

#define N 10

void help(const char *command_name){
    std::cerr << command_name << " <std::model>" << std::endl;
}


int main(int argc, char *argv[]){
    if(argc!=2){
        std::cerr << "Invalid number of arguments" << std::endl;
        help(argv[0]);
        return(1);
    }

    const char *csv_model = argv[1];

    std::set<std::vector<uint8_t>> hash_table;
    std::list<std::vector<uint8_t>> mdl;
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
    hash_table.insert(mdl.begin(), mdl.end());

    std::cout << hash_table.size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for(auto &i : mdl){
        auto r = hash_table.find(i);
        if(r == hash_table.end()){
            std::cout << "Not in set" << std::endl;
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Total duration: " << duration.count() << "ms [" << mdl.size() << "]" << std::endl;

}
