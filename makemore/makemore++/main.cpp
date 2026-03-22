#include "include/makemore.h"
#include <iostream>
#include <string>
#include <string_view>

using namespace makemore;

static void print_usage(std::string_view prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
         << "\n"
         << "  -i, --input-file    <path>   Input file, one word per line  [names.txt]\n"
         << "  -o, --work-dir      <path>   Directory for checkpoints      [out]\n"
         << "  -b, --batch-size    <int>    Mini-batch size                [32]\n"
         << "      --max-steps     <int>    Training steps (-1 = infinite) [-1]\n"
         << "      --top-k         <int>    Top-k sampling (-1 = off)      [-1]\n"
         << "  -l, --learning-rate <float>  Learning rate                  [5e-4]\n"
         << "      --seed          <uint>   RNG seed                       [3407]\n"
         << "      --resume                 Resume from checkpoint\n"
         << "      --sample-only            Sample from checkpoint and exit\n"
         << "  -h, --help                   Show this message\n";
}

static Config parse_args(int argc, char** argv)
{
    Config cfg;

    auto require_next = [&](int& i, std::string_view flag) -> std::string_view
    {
        if (i + 1 >= argc)
        {
            std::cerr << "error: " << flag << " requires an argument" << '\n';
            exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i)
    {
        std::string_view arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            exit(0);
        }

        if (arg == "-i" || arg == "--input-file")
        {
            cfg.input_file = require_next(i, arg);
        }
        else if (arg == "-o" || arg == "--work-dir")
        {
            cfg.work_dir = require_next(i, arg);
        }
        else if (arg == "-b" || arg == "--batch-size")
        {
            cfg.batch_size = std::stoi(std::string(require_next(i, arg)));
        }
        else if (arg == "--max-steps")
        {
            cfg.max_steps = std::stoi(std::string(require_next(i, arg)));
        }
        else if (arg == "--top-k")
        {
            cfg.top_k = std::stoi(std::string(require_next(i, arg)));
        }
        else if (arg == "-l" || arg == "--learning-rate")
        {
            cfg.learning_rate = std::stof(std::string(require_next(i, arg)));
        }
        else if (arg == "--seed")
        {
            cfg.seed = static_cast<unsigned int>(std::stoul(std::string(require_next(i, arg))));
        }
        else if (arg == "--resume")
        {
            cfg.resume = true;
        }
        else if (arg == "--sample-only")
        {
            cfg.sample_only = true;
        }
        else
        {
            std::cerr << "error: unknown argument '" << arg << "'" << '\n';
            print_usage(argv[0]);
            exit(1);
        }
    }

    return cfg;
}

int main(int argc, char** argv)
{
    const Config cfg = parse_args(argc, argv);

    std::cout << "makemore — bigram model (C++23)\n";
    std::cout << "input       : " << cfg.input_file.string() << '\n';
    std::cout << "work dir    : " << cfg.work_dir.string() << '\n';
    std::cout << "batch size  : " << cfg.batch_size << '\n';
    std::cout << "max steps   : " << cfg.max_steps << '\n';
    std::cout << "lr          : " << cfg.learning_rate << '\n';
    std::cout << "seed        : " << cfg.seed << "\n\n";

    try
    {
        train(cfg);
    }
    catch (const std::exception& e)
    {
        std::cerr << "fatal: " << e.what() << '\n';
        return 1;
    }

    return 0;
}