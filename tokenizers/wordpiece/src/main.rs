mod tokenizer;


fn main() {

    let vocab_size = 70;
    let save_path = String::from("token");
    let load_path: Vec<String> = vec!["/Users/adityamakkar/Desktop/CS/tokenizer/test.txt".to_string()];
    let mut test = tokenizer::Tokenizer::construct(
        vocab_size,
        save_path,
        load_path
    );

    test.train();

}
