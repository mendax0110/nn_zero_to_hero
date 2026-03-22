#pragma once

#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <span>

namespace makemore
{
    struct Config
    {
        std::filesystem::path input_file{"names.txt"};
        std::filesystem::path work_dir{"out"};
        int batch_size{32};
        int max_steps{-1};
        int top_k{-1};
        float learning_rate{5e-4f};
        unsigned int seed{3407};
        bool sample_only{false};
        bool resume{false};
    };

    class CharDataset
    {
    public:
        CharDataset(std::vector<std::string> words, std::vector<char> chars, int max_word_length);

        [[nodiscard]] int vocab_size() const;
        [[nodiscard]] int output_length() const;
        [[nodiscard]] int size() const;
        [[nodiscard]] bool contains(const std::string& word) const;

        [[nodiscard]] std::vector<int> encode(const std::string& word) const;
        [[nodiscard]] std::string decode(std::span<const int> indices) const;

        struct Sample { std::vector<int> x, y; };
        [[nodiscard]] Sample get(int idx) const;

        [[nodiscard]] const std::vector<std::string>& words() const
        {
            return words_;
        }

    private:
        std::vector<std::string> words_;
        std::vector<char> chars_;
        int max_word_length_;
        std::unordered_map<char, int> stoi_;
        std::unordered_map<int, char> itos_;
    };

    struct DatasetPair
    {
        CharDataset train;
        CharDataset test;
    };

    [[nodiscard]] DatasetPair create_datasets(const std::filesystem::path& file_input);

    class Bigram
    {
    public:
        explicit Bigram(int vocab_size);

        [[nodiscard]] std::pair<std::vector<float>, float> forward(std::span<const int> x, int batch, int seq, std::optional<std::span<const int>> targets = std::nullopt) const;

        void backward(std::span<const int> x, int batch, int seq, std::span<const int> targets);

        void update(float learning_rate);

        void zero_grad();

        [[nodiscard]] int vocab_size() const { return vocab_size_; }
        [[nodiscard]] int num_params() const { return vocab_size_ * vocab_size_; }

        void save(std::ostream& os) const;
        void load(std::istream& is);

    private:
        int vocab_size_;
        std::vector<float> logits_;
        std::vector<float> grad_;
    };

    [[nodiscard]] std::vector<int> generate(
                                    const Bigram& model, std::span<const int> init,
                                    int max_new_tokens, float temperature = 1.0f,
                                    bool do_sample = true, int top_k = -1,
                                    std::mt19937& rng = *[]() -> std::mt19937*
                                    {
                                        static std::mt19937 g{42}; return &g;
                                    }());

    void train(const Config& cfg);
}