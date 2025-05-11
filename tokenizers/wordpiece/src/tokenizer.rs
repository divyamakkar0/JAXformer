use std::collections::{HashMap, BTreeSet};
use std::fs;
use std::io::{BufWriter, Write};


#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Hash)]
struct Token {
    start: bool,
    token: String,
}

pub struct Tokenizer {
    pub vocab_size: u32,
    pub save_path: String,
    read_path: Vec<String>,
    global_vocab: BTreeSet<Token>,
    global_token_count: HashMap<Token, u32>,
    global_pair_count: HashMap<(Token, Token), u32>,
    global_tokens: Vec<Vec<Vec<Token>>>
}


impl Tokenizer {

    pub fn construct(
        vocab_size: u32,
        save_path: String,
        read_path: Vec<String>
    )  -> Self {
        Self {
            vocab_size: vocab_size,
            save_path: save_path,
            read_path: read_path,
            global_vocab: BTreeSet::new(),
            global_pair_count: HashMap::new(),
            global_token_count: HashMap::new(),
            global_tokens: Vec::new(),
        }
    }

    pub fn train(&mut self){
        for i in 0..self.read_path.len() {
            self.open_file_and_normalize(i );
        }
        self.print_tokens();

        while self.global_vocab.len() < self.vocab_size as usize {

            self.global_pair_count.clear();
            self.global_token_count.clear();

            for i in 0..self.read_path.len() {
                self.compute_stat(i);
            }
            let max_pair = self.find_max_pair();
            let new_token = Token{
                start: max_pair.0.start,
                token: format!("{}{}", max_pair.0.token, max_pair.1.token)
            };
            self.global_vocab.insert(new_token);
            for i in 0..self.read_path.len() {
                self.make_merge(i, max_pair.clone());
            }

        }

        self.print_vocab();
        self.save_vocab();

    }

    fn print_tokens(&self) {
        for i in 0..self.global_tokens.len() {
            for j in 0..self.global_tokens[i].len() {
                for k in 0..self.global_tokens[i][j].len() {
                    self.print_token(&self.global_tokens[i][j][k]);
                }
                println!();
            }
        }
    }

    fn print_token(&self, token: &Token) {
        let mut token_str: String = token.token.clone();
        if !token.start {token_str = "##".to_string() + &token_str};
        print!("{} ", token_str);
    }

    fn print_vocab(&self) {
        for (_, token) in self.global_vocab.iter().enumerate() {
            self.print_token(token);
            print!("\n");
       }
    }

    fn check_word(&self, c: &str) -> bool {
        let c_char = c
                                    .chars()
                                    .next()
                                    .expect("expected a char got None");

        if c_char.is_whitespace() || c_char == '\n' {return false};
        true
    }

    fn save_vocab(&self) {
        let file_res = fs::File::open(&self.save_path);
        let file = match file_res {
            Ok(_) =>  {
                fs::remove_file(&self.save_path).expect("Unable to remove file");
                fs::File::create(&self.save_path).expect("Unable to create file")
            },
            Err(_) => {
                fs::File::create(&self.save_path).expect("Unable to create file")
            }
        };

        let mut writer = BufWriter::new(file);
        for (_, token) in self.global_vocab.iter().enumerate() {
            let mut token_str: String = token.token.clone();
            if !token.start {token_str = "##".to_string() + &token_str};
            writeln!(writer, "{}", token_str).expect("Unable to write data");
        }
        writer.flush().expect("Unable to flush data");
    }

    fn find_max_pair(&self) -> (Token, Token) {
        let mut max_pair: (Token, Token) = (Token {start: false, token: "".to_string()}, Token {start: false, token: "".to_string()});
        let mut max_score: f32 = 0.0;


        for (key, value) in &self.global_pair_count {
            let score = *value as f32 / (self.global_token_count.get(&key.0).expect("expected key") * self.global_token_count.get(&key.1).expect("expected key")) as f32;
            if score >= max_score {
                max_score = score;
                max_pair = key.clone();
            }
        }
        max_pair
    }

    fn open_file_and_normalize(&mut self, idx: usize) {

        let contents = fs::read_to_string(&self.read_path[idx])
            .expect("Unable to open file");

        let mut words: Vec<&str> = Vec::new();

        let mut idx = 0;
        for (i,c) in contents.char_indices(){
            if !c.is_alphanumeric(){
                if idx + 1 < i {
                    words.push(&contents[idx..i]);
                }
                if self.check_word(&contents[i..i+1]) {
                    words.push(&contents[i..i+1])
                }
                idx = i + 1;
            }
        }

        if idx < contents.len() {
            words.push(&contents[idx..]);
        }

        let mut tokens: Vec<Vec<Token>> = Vec::new();

        for word in words {
            let mut word_tokens: Vec<Token> = Vec::new();
            for (i, c) in word.char_indices(){
                let new_token: Token = Token {
                    token: c.to_string(),
                    start: i == 0
                };
                word_tokens.push(new_token.clone());
                self.global_vocab.insert(new_token.clone());
            }
            tokens.push(word_tokens);
        }
        self.global_tokens.push(tokens);

    }

    fn compute_stat(&mut self, idx: usize) {

        let mut pair_stat: HashMap<(Token, Token), u32> = HashMap::new();
        let mut token_stat: HashMap<Token, u32> = HashMap::new();
        let tokens = &self.global_tokens[idx];

        let word_length = tokens.len();

        for i in 0..word_length {
            let length = tokens[i].len();
            for j in 0..length-1{
                let token_pair: (&Token, &Token) = (&tokens[i][j], &tokens[i][j+1]);

                pair_stat.entry((token_pair.0.clone(), token_pair.1.clone()))
                    .and_modify(|e| *e += 1)
                    .or_insert(1);

                token_stat.entry(token_pair.0.clone())
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
            }
            token_stat.entry(tokens[i][length-1].clone())
                .and_modify(|e| *e += 1)
                .or_insert(1);

        }

        for (key, value) in &pair_stat {
            self.global_pair_count.entry(key.clone())
                .and_modify(|e| *e += value)
                .or_insert(*value);
        }

        for (key, value) in &token_stat {
            self.global_token_count.entry(key.clone())
                .and_modify(|e| *e += value)
                .or_insert(*value);
        }

    }

    fn make_merge(&mut self, idx:usize, pair: (Token, Token)) {

        let tokens = &self.global_tokens[idx];
        let mut new_tokens: Vec<Vec<Token>> = Vec::new();


        for i in 0..tokens.len() {
            let mut new_word: Vec<Token> = Vec::new();
            let length = tokens[i].len();
            let mut j = 0;
            while j < length {
                if j < length - 1 && tokens[i][j] == pair.0 && tokens[i][j + 1] == pair.1 {
                    let new_string = format!("{}{}", pair.0.token, pair.1.token);
                    let new_token = Token {
                        token: new_string,
                        start: pair.0.start,
                    };
                    new_word.push(new_token);
                    j += 2;
                } else {
                    new_word.push(tokens[i][j].clone());
                    j += 1;
                }
            }
            new_tokens.push(new_word);
        }
        self.global_tokens[idx] = new_tokens;

    }


}
