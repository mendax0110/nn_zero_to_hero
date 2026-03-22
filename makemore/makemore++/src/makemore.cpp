#include "../include/makemore.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>

using namespace makemore;

CharDataset::CharDataset(std::vector<std::string> words, std::vector<char> chars, const int max_word_length)
    : words_(std::move(words))
    , chars_(std::move(chars))
    , max_word_length_(max_word_length)
{
    for (int i = 0; i < static_cast<int>(chars_.size()); ++i)
    {
        stoi_[chars_[i]] = i + 1;
        itos_[i + 1] = chars_[i];
    }
}

int CharDataset::vocab_size() const
{
    return static_cast<int>(chars_.size()) + 1;
}

int CharDataset::output_length() const
{
    return max_word_length_ + 1;
}

int CharDataset::size() const
{
    return static_cast<int>(words_.size());
}

bool CharDataset::contains(const std::string& word) const
{
    return std::ranges::find(words_, word) != words_.end();
}

std::vector<int> CharDataset::encode(const std::string& word) const
{
    std::vector<int> out;
    out.reserve(word.size());
    for (char c : word)
    {
        auto it = stoi_.find(c);
        if (it == stoi_.end())
        {
            throw std::runtime_error(std::string("Unknown character: ") + c);
        }
        out.push_back(it->second);
    }
    return out;
}

std::string CharDataset::decode(const std::span<const int> indices) const
{
    std::string out;
    out.reserve(indices.size());
    for (int idx : indices)
    {
        if (auto it = itos_.find(idx); it != itos_.end())
        {
            out += it->second;
        }
    }
    return out;
}

CharDataset::Sample CharDataset::get(const int idx) const
{
    const std::string& word = words_[idx];
    const auto ix = encode(word);
    const int len = output_length();

    Sample s;
    s.x.assign(len, 0);
    s.y.assign(len, -1);

    // input sequence: <START> followed by characters
    s.x[0] = 0;
    for (int i = 0; i < static_cast<int>(ix.size()); ++i)
    {
        s.x[i + 1] = ix[i];
    }

    // targets: first char after <START>, then next chars, then <STOP>
    if (!ix.empty())
    {
        s.y[0] = ix[0];
        for (int i = 0; i < static_cast<int>(ix.size()) - 1; ++i)
        {
            s.y[i + 1] = ix[i + 1];
        }
        if (static_cast<int>(ix.size()) < len)
        {
            s.y[static_cast<int>(ix.size())] = 0;
        }
    }
    else
    {
        s.y[0] = 0;
    }

    return s;
}

DatasetPair makemore::create_datasets(const std::filesystem::path& file_input)
{
    std::ifstream f(file_input);
    if (!f.good())
    {
        throw std::runtime_error("Can't open input file: " + file_input.string());
    }

    std::vector<std::string> words;
    std::string line;
    while (std::getline(f, line))
    {
        const auto first = line.find_first_not_of(" \t\r\n");
        const auto last = line.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        words.push_back(line.substr(first, last - first + 1));
    }

    std::set<char> char_set;
    for (const auto& w : words)
    {
        for (char c : w)
        {
            char_set.insert(c);
        }
    }

    std::vector<char> chars(char_set.begin(), char_set.end());

    int max_len = 0;
    for (const auto& w : words)
    {
        max_len = std::max(max_len, static_cast<int>(w.size()));
    }

    std::cout << "examples: " << words.size() << " max_word_length: " << max_len
              << " vocab: " << chars.size() + 1 << " chars: " << std::string(chars.begin(), chars.end()) << '\n';
    std::mt19937 rng(42);
    std::ranges::shuffle(words, rng);

    const int test_size = std::min(1000, static_cast<int>(words.size()) / 10);
    const int train_size = static_cast<int>(words.size()) - test_size;

    std::vector<std::string> train_words(words.begin(), words.begin() + train_size);
    std::vector<std::string> test_words(words.begin() + train_size, words.end());

    std::cout << "train: " << train_words.size() << "  test: " << test_words.size() << '\n';

    return DatasetPair{ CharDataset(std::move(train_words), chars, max_len), CharDataset(std::move(test_words), chars, max_len) };
}

Bigram::Bigram(const int vocab_size)
    : vocab_size_(vocab_size)
    , logits_(vocab_size * vocab_size, 0.0f)
    , grad_(vocab_size * vocab_size, 0.0f)
{

}

std::pair<std::vector<float>, float> Bigram::forward(std::span<const int> x, int batch, int seq, std::optional<std::span<const int>> targets) const
{
    const int V = vocab_size_;
    const int N = batch * seq;

    std::vector<float> out(N * V);
    for (int i = 0; i < N; ++i)
    {
        const int tok = x[i];
        std::copy(logits_.data() + tok * V, logits_.data() + tok * V + V, out.data() + i * V);
    }

    float loss = 0.0f;
    if (targets)
    {
        int count = 0;
        for (int i = 0; i < N; ++i)
        {
            int tgt = (*targets)[i];
            if (tgt == -1) continue;

            const float* row = out.data() + i * V;
            float max_v = *std::max_element(row, row + V);
            float sum_e = 0.0f;
            for (int v = 0; v < V; ++v)
            {
                sum_e += std::exp(row[v] - max_v);
            }
            loss += -(row[tgt] - max_v - std::log(sum_e));
            ++count;
        }
        if (count > 0) loss /= static_cast<float>(count);
    }

    return std::make_pair(out, loss);
}

void Bigram::backward(std::span<const int> x, int batch, int seq, std::span<const int> targets)
{
    const int V = vocab_size_;
    const int N = batch * seq;

    auto [logits_out_, _] = forward(x, batch, seq, std::nullopt);

    int count = 0;
    for (int i = 0; i < N; ++i)
    {
        int tgt = targets[i];
        if (tgt == -1) continue;
        ++count;

        int tok = x[i];
        const float* row = logits_out_.data() + i * V;

        float max_v = *std::max_element(row, row + V);
        float sum_e = 0.0f;
        for (int v = 0; v < V; ++v)
        {
            sum_e += std::exp(row[v] - max_v);
        }

        float* g = grad_.data() + tok * V;
        for (int v = 0; v < V; ++v)
        {
            float sm = std::exp(row[v] - max_v) / sum_e;
            g[v] += sm - (v == tgt ? 1.0f : 0.0f);
        }
    }

    if (count > 0)
    {
        float inv = 1.0f / static_cast<float>(count);
        for (auto& g : grad_)
        {
            g *= inv;
        }
    }
}

void Bigram::update(float learning_rate)
{
    for (int i = 0; i < static_cast<int>(logits_.size()); ++i)
    {
        logits_[i] -= learning_rate * grad_[i];
    }
}

void Bigram::zero_grad()
{
    std::ranges::fill(grad_, 0.0f);
}

void Bigram::save(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(logits_.data()), logits_.size() * sizeof(float));
}

void Bigram::load(std::istream& is)
{
    is.read(reinterpret_cast<char*>(logits_.data()), logits_.size() * sizeof(float));
}

std::vector<int> makemore::generate(const Bigram& model, std::span<const int> init, int max_new_tokens, float temperature, bool do_sample, int top_k, std::mt19937& rng)
{
    std::vector<int> idx(init.begin(), init.end());
    const int V = model.vocab_size();

    for (int step = 0; step < max_new_tokens; ++step)
    {
        std::vector<int> ctx = { idx.back() };

        auto [logits, _] = model.forward(ctx, 1, 1, std::nullopt);

        for (auto& l : logits)
        {
            l /= temperature;
        }

        if (top_k > 0 && top_k < V)
        {
            std::vector<float> sorted_l = logits;
            std::partial_sort(sorted_l.begin(), sorted_l.begin() + top_k, sorted_l.end(), std::greater<float>{});
            const float kth = sorted_l[top_k - 1];
            for (auto& l : logits)
            {
                if (l < kth)
                {
                    l = -1e9f;
                }
            }
        }

        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_l = 0.0f;

        for (auto& l : logits)
        {
            l = std::exp(l - max_l);
            sum_l += l;
        }

        for (auto& l : logits)
        {
            l /= sum_l;
        }

        int next;
        if (do_sample)
        {
            std::discrete_distribution<int> dist(logits.begin(), logits.end());
            next = dist(rng);
        }
        else
        {
            next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }

        idx.push_back(next);
        if (next == 0) break;
    }

    return idx;
}

static void print_samples(const Bigram& model, const CharDataset& train_ds, const CharDataset& test_ds, int top_k, int num = 10)
{
    std::mt19937 rng(0);
    const int max_len = train_ds.output_length() - 1;

    std::vector<std::string> in_train, in_test, novel;

    const int sample_top_k = top_k > 0 ? top_k : 20; // default to a modest top-k for higher-quality samples
    const float temperature = 0.8f;                  // slightly cooled sampling to reduce gibberish

    for (int i = 0; i < num; ++i)
    {
        std::vector<int> init  = { 0 };   // start with <START> token
        auto samp  = makemore::generate(model, init, max_len, temperature, true, sample_top_k, rng);

        // drop leading <START>, trailing <STOP>
        std::vector<int> tokens(samp.begin() + 1, samp.end());
        if (!tokens.empty() && tokens.back() == 0)
            tokens.pop_back();

        std::string word = train_ds.decode(tokens);

        if (train_ds.contains(word))
        {
            in_train.push_back(word);
        }
        else if (test_ds .contains(word))
        {
            in_test.push_back(word);
        }
        else
        {
            novel.push_back(word);
        }
    }

    std::cout << std::string(80, '-') << '\n';
    for (auto&& [lst, label] : std::vector<std::pair<std::vector<std::string>, std::string_view>>
        { {in_train, "in train"}, {in_test, "in test"}, {novel, "new"} })
    {
        std::cout << lst.size() << " samples that are " << label << ":";
        for (const auto& w : lst) std::cout << "  " << w;
        std::cout << '\n';
    }
    std::cout << std::string(80, '-') << '\n';
}

void makemore::train(const Config& cfg)
{
    auto [train_ds, test_ds] = create_datasets(cfg.input_file);

    Bigram model(train_ds.vocab_size());
    //std::cout << "bigram model  params: " << model.num_params() << '\n';

    std::filesystem::create_directories(cfg.work_dir);
    std::filesystem::path ckpt = cfg.work_dir / "model.bin";

    if (cfg.resume || cfg.sample_only)
    {
        if (std::ifstream ifs(ckpt, std::ios::binary); ifs.good())
        {
            model.load(ifs);
            // std::cout << "resumed from " << ckpt.string() << '\n';
        }
    }

    if (cfg.sample_only)
    {
        print_samples(model, train_ds, test_ds, cfg.top_k, 50);
        return;
    }

    std::mt19937 data_rng(cfg.seed);
    std::uniform_int_distribution<int> idx_dist(0, train_ds.size() - 1);
    std::uniform_int_distribution<int> test_dist(0, test_ds.size() -1);

    const int seq = train_ds.output_length();
    float best_loss = std::numeric_limits<float>::infinity();

    for (int step = 0; cfg.max_steps < 0 || step < cfg.max_steps; ++step)
    {
        std::vector<int> x_batch(cfg.batch_size * seq);
        std::vector<int> y_batch(cfg.batch_size * seq);

        for (int b = 0; b < cfg.batch_size; ++b)
        {
            auto [x, y] = train_ds.get(idx_dist(data_rng));
            std::copy(x.begin(), x.end(), x_batch.data() + b * seq);
            std::copy(y.begin(), y.end(), y_batch.data() + b * seq);
        }

        auto t0 = std::chrono::steady_clock::now();

        auto [logits, loss] = model.forward(x_batch, cfg.batch_size, seq, y_batch);

        model.zero_grad();
        model.backward(x_batch, cfg.batch_size, seq, y_batch);
        model.update(cfg.learning_rate);

        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();

        if (step % 10 == 0)
        {
            std::cout << "step " << std::setw(5) << step << " | loss " << std::fixed << std::setprecision(4) << loss << " | " << ms << "ms" << '\n';
        }

        if (step > 0 && step % 500 == 0)
        {
            float test_loss = 0.0f;
            for (int eb = 0; eb < 10; ++eb)
            {
                std::vector<int> tx(cfg.batch_size * seq), ty(cfg.batch_size * seq);
                for (int b = 0; b < cfg.batch_size; ++b)
                {
                    auto [x, y] = test_ds.get(test_dist(data_rng));
                    std::copy(x.begin(), x.end(), tx.data() + b * seq);
                    std::copy(y.begin(), y.end(), ty.data() + b * seq);
                }
                auto [_, l] = model.forward(tx, cfg.batch_size, seq, ty);
                test_loss += l;
            }
            test_loss /= 10.0f;
            std::cout << "step " << std::setw(5) << step << " | test loss " << std::fixed << std::setprecision(4) << test_loss << '\n';

            if (test_loss < best_loss)
            {
                best_loss = test_loss;
                std::ofstream ofs(ckpt, std::ios::binary);
                model.save(ofs);
                std::cout << "checkpoint -> " << ckpt.string() << '\n';
            }
        }

        if (step > 0 && step % 200 == 0)
        {
            print_samples(model, train_ds, test_ds, cfg.top_k, 10);
        }
    }
}
