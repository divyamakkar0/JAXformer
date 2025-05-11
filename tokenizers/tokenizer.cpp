#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <thread>
#include <mutex>
#include <chrono>


using namespace std;
using namespace std::chrono;

mutex mtx;

struct Token {
  string token;
  bool start;

  Token(string a, int pos){
    this->token = a;
    this->start = (pos == 0);
    if(a.size() == 1 && !isalpha(a[0])) this->start = true;
  }

  void print() const {
    cout << (!start ? "##" : "") + token << endl;
  }

  bool operator== ( const Token &b) const {
    return (this->start == b.start && this->token == b.token);
  }

};


struct TokenCompare {
  bool operator() (const Token &a, const Token &b) const {
    if (a.start != b.start) return a.start;
    return a.token < b.token;
  }
};

struct TokenComparePair {
  bool operator() (const pair<Token, Token> &a, const pair<Token, Token> &b) const{
    TokenCompare cmp;
    if(cmp(a.first, b.first)) return true;
    if(cmp(b.first, a.first)) return false;
    return cmp(a.second, b.second);
  }
};

class Tokenizer {

  public:

    Tokenizer(int vocab_size, vector<string> &v, string load_path) :
    vocab_size(vocab_size), file_names(v), save_path(load_path) {}


    void train(){

      vector<thread> threads;
      cout << "tokenizing ..." << endl;
      for(int i = 0; i < (int) file_names.size(); ++i){
        threads.push_back(thread(&Tokenizer::open_file_and_tokenize, this, file_names[i]));
      }

      for(auto &t: threads) t.join();
      threads.clear();
      cout << "computing & merging ..." << endl;

      while(global_vocab.size() < vocab_size){

        cout << "vocab size: " << global_vocab.size() << endl;
        global_pair_statistics.clear();
        global_token_statistics.clear();

        for(int i = 0; i < (int) global_tokens.size(); ++i){
          threads.push_back(thread(&Tokenizer::compute_stat, this, i));
        }

        for(auto &t: threads) t.join();
        threads.clear();

        map<pair<Token, Token>, float, TokenComparePair> scores;
        for(auto [key, value]: global_pair_statistics){
          scores[key] =  (value * 1.0) / (1.0 * global_token_statistics[key.first] * global_token_statistics[key.second]);
        }

        pair<Token,Token> merge_to_make = (*scores.begin()).first;
        float max_score = (*scores.begin()).second;

        for(auto [k, v]: scores){
          if(v > max_score){
            max_score = v;
            merge_to_make = k;
          }
        }
        Token new_token = Token(
          merge_to_make.first.token + merge_to_make.second.token,
          merge_to_make.first.start ? 0 : 1
        );
        new_token.print();

        global_vocab.insert(new_token);
        for(int i = 0; i < (int) global_tokens.size(); ++i){
          threads.push_back(thread(&Tokenizer::merge_tokens, this, merge_to_make, i));
        }
        for(auto &t: threads) t.join();
        threads.clear();

      }

      this->print_out_vocab();
    };

    void open_file_and_tokenize(
      string path
    ){

      ifstream file(path, std::ios::binary);
      vector<string> file_content;
      string line;

      while(getline(file, line)) {
        int str_start = 0;
        for(int i = 0; i < line.size(); ++i){
          if(!isalnum(line[i])) {
            if(str_start + 1 < i) file_content.push_back(line.substr(str_start, i - str_start));
            if(!char_to_exclude.count(line[i]) ) file_content.push_back(line.substr(i, 1));
            str_start = i + 1;
          }
        }
        if(str_start < line.size()) file_content.push_back(line.substr(str_start, line.size() - str_start));
      }
      file.close();

      vector<vector<Token>> tokens;
      set<Token, TokenCompare> local_vocab;

      for(string s: file_content){
        vector<Token> current_word;
        for(int i = 0; i < (int)s.size(); ++i){
          Token c(s.substr(i, 1), i);
          local_vocab.insert(c);
          current_word.push_back(c);
        }

        tokens.push_back(current_word);
      }


      mtx.lock();

      for(Token i: local_vocab) global_vocab.insert(i);
      global_tokens.push_back(tokens);

      mtx.unlock();

      return;
    }

    void compute_stat(
      int idx
    ){

      map<pair<Token, Token>, int, TokenComparePair> pair_statistics;
      map<Token, int, TokenCompare> token_statistics;
      vector<vector<Token>> &tokens = global_tokens[idx];
      int word_length = (int) tokens.size();

      for(int i = 0; i < word_length; ++i){
        int length = (int) tokens[i].size();
        for(int k = 0; k < length - 1; ++k){
          pair<Token, Token> current_pair = make_pair(tokens[i][k], tokens[i][k+1]);
          ++pair_statistics[current_pair];
          ++token_statistics[tokens[i][k]];
        }
        ++token_statistics[tokens[i][length - 1]];
      }

      mtx.lock();
      for(auto [key, value]: pair_statistics){
        global_pair_statistics[key] += value;
      }
      for(auto [key, value]: token_statistics){
        global_token_statistics[key] += value;
      }
      mtx.unlock();

    }


    void merge_tokens(
      pair<Token, Token> merge,
      int idx
    ){

      vector<vector<Token>> &tokens = global_tokens[idx];
      int n_tokens = (int) tokens.size();
      for(int i = 0; i < n_tokens; ++i){

        int length = (int)tokens[i].size();
        vector<int> del_idx;
        for(int k = 0; k < length - 1; ++k){
          if(merge.first == tokens[i][k] && merge.second == tokens[i][k+1]){
            tokens[i][k] = Token(tokens[i][k].token + tokens[i][k+1].token, k);
            del_idx.push_back(++k);
          }

        }

        for(int k = 0; k < (int)del_idx.size(); ++k){
          int idx = del_idx[k] - k;
          tokens[i].erase(tokens[i].begin() + idx);
        }

      }

      return;

    }


    void print_out_vocab(){
      for(Token i: global_vocab) i.print();
    }

    void save() {
      ofstream file(save_path);
      for(Token i: global_vocab) {
        file << i.token << " " << i.start << endl;
      }
      file.close();
    };

    void load(string file_path) {};
    vector<int> encode(string str);
    string decode(vector<int> &v);


  private:
    vector<vector<vector<Token>>> global_tokens;
    map<pair<Token, Token>, int, TokenComparePair> global_pair_statistics;
    map<Token, int, TokenCompare> global_token_statistics;

    set<Token, TokenCompare> global_vocab;


    set<char> char_to_exclude = {'\n', ' '};
    vector<string> file_names;
    string save_path;
    int vocab_size;

};

int main(){

  vector<string> path_name = {
    "tests/1.txt",
    "tests/2.txt",
    "tests/3.txt",
    "tests/4.txt",
  };

  for(int i = 1; i <= 2; ++i){
    string path = "training-monolingual.tokenized.shuffled/news.en-00";
    string num = to_string(i);
    while(num.size() < 3) num = "0" + num;
    path += num + "-of-00100";
    cout << path << endl;
    path_name.push_back(path);
  }


  Tokenizer test(65, path_name, "vocab.txt");
  auto start = high_resolution_clock::now();
  test.train();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << (duration.count() *1.0) / 1000000 << " seconds" << endl;

  test.save();
}
