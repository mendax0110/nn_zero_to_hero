#include "include/value.h"
#include "include/nn.h"
#include "include/draw_dot.h"

#include <iostream>
#include <vector>
#include <format>
#include <cmath>
#include <iomanip>
#include <print>

void section(std::string_view title)
{
    std::cout << "\n============================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "============================================================\n";
}

void demo_numerical_derivative()
{
    section("1. Numerical derivative  f(x) = 3x^2 - 4x + 5");

    auto f = [](double x) { return 3*x*x - 4*x + 5; };
    constexpr double h = 1e-8;

    for (double x : {3.0, -3.0, 2.0/3.0})
    {
        double deriv = (f(x + h) - f(x)) / h;
        std::println("  f'({:6.4f}) ~= {:.6f}   (exact: {})", x, deriv, 6*x - 4);
    }
}

void demo_manual_gradient()
{
    section("2. Manual gradient  d = a*b + c");

    auto a= make_val(2.0,  "a");
    auto b= make_val(-3.0, "b");
    auto c= make_val(10.0, "c");
    auto d= a * b + c;  d->label = "d";

    d->backward();

    std::println("  d = {:.2f}", d->data);
    std::println("  dd/da = {:.2f}  (expected: {:.2f})", a->grad, b->data);
    std::println("  dd/db = {:.2f}  (expected: {:.2f})", b->grad, a->data);
    std::println("  dd/dc = {:.2f}  (expected: 1.00)",   c->grad);
}

void demo_full_graph()
{
    section("3. Full graph  L = (a*b + c) * f  -- auto backward()");

    auto a= make_val(2.0,  "a");
    auto b= make_val(-3.0, "b");
    auto c= make_val(10.0, "c");
    auto e= a * b;  e->label = "e";
    auto d= e + c;  d->label = "d";
    auto f= make_val(-2.0, "f");
    auto L= d * f;  L->label = "L";

    draw_dot(L, "graph_L_before_backward");

    L->backward();

    std::println("  L     = {:.4f}", L->data);
    std::println("  dL/da = {:.4f}  (expected:  6.0000)", a->grad);
    std::println("  dL/db = {:.4f}  (expected: -4.0000)", b->grad);
    std::println("  dL/dc = {:.4f}  (expected: -2.0000)", c->grad);
    std::println("  dL/df = {:.4f}  (expected:  4.0000)", f->grad);

    draw_dot(L, "graph_L_after_backward");
}

void demo_gradient_step()
{
    section("4. Gradient descent step  delta = 0.01 * grad");

    auto a= make_val(2.0,  "a");
    auto b= make_val(-3.0, "b");
    auto c= make_val(10.0, "c");
    auto f= make_val(-2.0, "f");

    auto compute_L = [&] { return (a * b + c) * f; };

    auto L0 = compute_L();  L0->backward();
    std::println("  L before step: {:.6f}", L0->data);

    constexpr double lr = 0.01;
    a->data += lr * a->grad;
    b->data += lr * b->grad;
    c->data += lr * c->grad;
    f->data += lr * f->grad;

    auto L1 = compute_L();
    std::println("  L after  step: {:.6f}  (should increase toward 0)", L1->data);
}

void demo_single_neuron()
{
    section("5. Single neuron  o = tanh(x1*w1 + x2*w2 + b)");

    auto x1= make_val(2.0,  "x1");
    auto x2= make_val(0.0,  "x2");
    auto w1= make_val(-3.0, "w1");
    auto w2= make_val(1.0,  "w2");
    auto b= make_val(6.881375870195432, "b");

    auto o= tanh(x1*w1 + x2*w2 + b);  o->label = "o";
    draw_dot(o, "graph_neuron_before_backward");
    o->backward();

    std::println("  o      = {:.6f}  (expected:  0.707107)", o->data);
    std::println("  do/dx1 = {:.6f}  (expected: -1.500000)", x1->grad);
    std::println("  do/dx2 = {:.6f}  (expected:  0.500000)", x2->grad);
    std::println("  do/dw1 = {:.6f}  (expected:  1.000000)", w1->grad);
    std::println("  do/dw2 = {:.6f}  (expected:  0.000000)", w2->grad);

    draw_dot(o, "graph_neuron_after_backward");
}

void demo_tanh_decomposed()
{
    section("6. tanh decomposed into exp / arithmetic");

    auto x1= make_val(2.0,  "x1");
    auto x2= make_val(0.0,  "x2");
    auto w1= make_val(-3.0, "w1");
    auto w2= make_val(1.0,  "w2");
    auto b= make_val(6.881375870195432, "b");

    auto n= x1*w1 + x2*w2 + b;
    auto e2n= exp(make_val(2.0) * n);
    auto o= (e2n - 1.0) / (e2n + 1.0);  o->label = "o";

    o->backward();
    draw_dot(o, "graph_tanh_decomposed");

    std::println("  o      = {:.6f}  (should match section 5)", o->data);
    std::println("  do/dx1 = {:.6f}  (should match section 5)", x1->grad);
    std::println("  do/dw1 = {:.6f}  (should match section 5)", w1->grad);
}

void demo_mlp_training()
{
    section("7. MLP 3->[4,4]->1  binary classifier  (20 epochs)");

    const std::vector<std::vector<double>> xs = {
            { 2.0,  3.0, -1.0},
            { 3.0, -1.0,  0.5},
            { 0.5,  1.0,  1.0},
            { 1.0,  1.0, -1.0},
    };
    const std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    MLP net{3, {4, 4, 1}};
    std::println("  Parameters: {}", net.param_count());

    draw_dot(net(std::span{xs[0]})[0], "graph_mlp_forward");

    constexpr double lr = 0.05;
    constexpr int epochs = 20;

    for (int k = 0; k < epochs; ++k)
    {
        std::vector<ValuePtr> ypred;
        for (const auto& x : xs)
        {
            ypred.push_back(net(std::span{x})[0]);
        }

        // MSE loss
        auto loss = make_val(0.0);
        for (std::size_t i = 0; i < ys.size(); ++i)
        {
            auto diff = ypred[i] - ys[i];
            loss = loss + diff * diff;
        }

        net.zero_grad();
        loss->backward();
        net.step(lr);

        std::println("  epoch {:2d}  loss = {:.6f}", k, loss->data);
    }

    std::println("\n  Final predictions vs targets:");
    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        auto pred = net(std::span{xs[i]})[0]->data;
        std::println("    sample {}: pred={:+.4f}  target={:+.1f}  {}", i, pred, ys[i], std::abs(pred - ys[i]) < 0.5 ? "OK" : "WRONG");
    }
}

int main()
{
    std::println("micrograd++");
    std::println("==========================================");

    demo_numerical_derivative();
    demo_manual_gradient();
    demo_full_graph();
    demo_gradient_step();
    demo_single_neuron();
    demo_tanh_decomposed();
    demo_mlp_training();

    std::println("\n============================================================");
    std::println("  All demos complete.");
    std::println("============================================================\n");
}

