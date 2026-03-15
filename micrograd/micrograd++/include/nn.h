#pragma once

#include "value.h"
#include <random>
#include <ranges>
#include <vector>
#include <span>
#include <numeric>

inline std::mt19937& rng()
{
    static std::mt19937 gen{std::random_device{}()};
    return gen;
}

inline double rand_uniform(double lo = -1.0, double hi = 1.0)
{
    return std::uniform_real_distribution<double>{lo, hi}(rng());
}

class Neuron
{
public:
    std::vector<ValuePtr> w;
    ValuePtr b;

    enum class Activation
    {
        Tanh,
        ReLU,
        Linear
    };

    Activation act_fn;

    explicit Neuron(size_t nin, Activation act = Activation::Tanh)
        : b{make_val(rand_uniform())}
        , act_fn{act}
    {
        w.reserve(nin);
        for ([[maybe_unused]] std::weakly_incrementable auto _ : std::views::iota(0uz, nin))
        {
            w.push_back(make_val(rand_uniform()));
        }
    }

    ValuePtr operator()(std::span<const ValuePtr> x) const
    {
        assert(x.size() == w.size());

        auto act = std::inner_product(
                w.begin(), w.end(), x.begin(), b,
                [](ValuePtr acc, ValuePtr term) { return acc + term; },
                [](ValuePtr wi, const ValuePtr& xi) { return wi * xi; });

        switch (act_fn)
        {
            case Activation::Tanh: return tanh(act);
            case Activation::ReLU: return relu(act);
            case Activation::Linear: return act;
        }

        return act;
    }

    [[nodiscard]] std::vector<ValuePtr> parameters() const
    {
        auto params = w;
        params.push_back(b);
        return params;
    }
};

class Layer
{
public:
    std::vector<Neuron> neurons;

    Layer(size_t nin, size_t nout, Neuron::Activation act = Neuron::Activation::Tanh)
    {
        neurons.reserve(nout);
        for ([[maybe_unused]] std::weakly_incrementable auto _ : std::views::iota(0uz, nout))
        {
            neurons.emplace_back(nin, act);
        }
    }


    std::vector<ValuePtr> operator()(std::span<const ValuePtr> x) const
    {
        std::vector<ValuePtr> outs;
        outs.reserve(neurons.size());
        for (const auto& neuron : neurons)
        {
            outs.push_back(neuron(x));
        }

        return outs;
    }

    [[nodiscard]] std::vector<ValuePtr> parameters() const
    {
        std::vector<ValuePtr> params;
        for (const auto& neuron : neurons)
        {
            for (auto& p : neuron.parameters())
            {
                params.push_back(p);
            }
        }

        return params;
    }
};

class MLP
{
public:
    std::vector<Layer> layers;

    MLP(size_t nin, std::vector<size_t> layer_sizes)
    {
        layers.reserve(layer_sizes.size());
        size_t in = nin;

        for (size_t i = 0; i < layer_sizes.size(); ++i)
        {
            auto act = (i + 1 == layer_sizes.size()) ? Neuron::Activation::Tanh : Neuron::Activation::Tanh;
            layers.emplace_back(in, layer_sizes[i], act);
            in = layer_sizes[i];
        }
    }

    std::vector<ValuePtr> operator()(std::span<const ValuePtr> x) const
    {
        std::vector<ValuePtr> current{x.begin(), x.end()};
        for (const auto& layer : layers)
        {
            current = layer(current);
        }
        return current;
    }

    std::vector<ValuePtr> operator()(std::span<const double> x) const
    {
        std::vector<ValuePtr> vals;
        vals.reserve(x.size());
        for (double v : x)
        {
            vals.push_back(make_val(v));
        }
        return (*this)(std::span{vals});
    }

    [[nodiscard]] std::vector<ValuePtr> parameters() const
    {
        std::vector<ValuePtr> params;
        for (const auto& layer : layers)
        {
            for (auto& p : layer.parameters())
            {
                params.push_back(p);
            }
        }

        return params;
    }

    [[nodiscard]] size_t param_count() const { return parameters().size(); }

    void zero_grad() const
    {
        for (auto& param : parameters())
        {
            param->grad = 0.0;
        }
    }

    void step(double lr) const
    {
        for (auto& param : parameters())
        {
            param->data -= lr * param->grad;
        }
    }
};