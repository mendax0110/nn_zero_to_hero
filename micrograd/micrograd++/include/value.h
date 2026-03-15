#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <format>
#include <cassert>

class Value;
using ValuePtr = std::shared_ptr<Value>;

template<typename... Args>
ValuePtr make_val(Args&&... args)
{
    return std::make_shared<Value>(std::forward<Args>(args)...);
}

class Value : public std::enable_shared_from_this<Value>
{
public:
    double data{};
    double grad{0.0};
    std::string label;

    std::string _op;
    std::vector<ValuePtr> _prev;
    std::function<void()> _backward{[] {}};

    explicit Value(double data, std::string label = "")
        : data{data}
        , label{std::move(label)}
    { }

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    friend ValuePtr operator+(ValuePtr a, ValuePtr b)
    {
        auto out = make_val(a->data + b->data, "");
        out->_op = "+";
        out->_prev = {a, b};
        out->_backward = [a, b, out]
        {
            a->grad += out->grad;
            b->grad += out->grad;
        };
        return out;
    }

    friend ValuePtr operator+(ValuePtr a, double b) { return a + make_val(b); }
    friend ValuePtr operator+(double a, ValuePtr b) { return make_val(a) + b; }

    friend ValuePtr operator*(ValuePtr a, ValuePtr b)
    {
        auto out = make_val(a->data * b->data, "");
        out->_op = "*";
        out->_prev = {a, b};
        out->_backward = [a, b, out]
        {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };
        return out;
    }

    friend ValuePtr operator*(ValuePtr a, double b) { return a * make_val(b); }
    friend ValuePtr operator*(double a, ValuePtr b) { return make_val(a) * b; }

    friend ValuePtr operator-(ValuePtr a, ValuePtr b) { return a + (b * -1.0); }
    friend ValuePtr operator-(ValuePtr a, double b) { return a + make_val(-b); }
    friend ValuePtr operator-(double a, ValuePtr b) { return make_val(a) + (b * -1.0); }
    friend ValuePtr operator-(ValuePtr a) { return a * -1.0; }

    friend ValuePtr pow(ValuePtr base, double exp)
    {
        auto out = make_val(std::pow(base->data, exp), "");
        out->_op = std::format("**{:.2g}", exp);
        out->_prev = {base};
        out->_backward = [base, exp, out]
        {
            base->grad += exp * std::pow(base->data, exp - 1.0) * out->grad;
        };
        return out;
    }

    friend ValuePtr operator/(ValuePtr a, ValuePtr b) { return a * pow(b, -1.0); }
    friend ValuePtr operator/(ValuePtr a, double b)   { return a * make_val(1.0 / b); }
    friend ValuePtr operator/(double a, ValuePtr b)   { return make_val(a) * pow(b, -1.0); }

    friend ValuePtr tanh(ValuePtr x)
    {
        double t = std::tanh(x->data);
        auto out = make_val(t, "");
        out->_op = "tanh";
        out->_prev = {x};
        out->_backward = [x, t, out]
        {
            x->grad += (1.0 - t * t) * out->grad;
        };
        return out;
    }

    friend ValuePtr relu(ValuePtr x)
    {
        double r = x->data > 0.0 ? x->data : 0.0;
        auto out = make_val(r, "");
        out->_op = "relu";
        out->_prev = {x};
        out->_backward = [x, out]
        {
            x->grad += (x->data > 0.0 ? 1.0 : 0.0) * out->grad;
        };
        return out;
    }

    friend ValuePtr exp(ValuePtr x)
    {
        double e = std::exp(x->data);
        auto out = make_val(e, "");
        out->_op = "exp";
        out->_prev = {x};
        out->_backward = [x, out]
        {
            x->grad += out->data * out->grad;
        };
        return out;
    }

    void backward()
    {
        std::vector<Value*> topo;
        std::set<Value*> visited;

        std::function<void(Value*)> build = [&](Value* v)
        {
            if (visited.contains(v)) return;
            visited.insert(v);
            for (auto& child : v->_prev)
            {
                build(child.get());
            }
            topo.push_back(v);
        };

        build(this);

        grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            (*it)->_backward();
        }
    }

    void zero_grad()
    {
        std::set<Value*> visited;
        std::function<void(Value*)> zero = [&](Value* v)
        {
            if (visited.contains(v)) return;
            visited.insert(v);
            v->grad = 0.0;
            for (auto& child : v->_prev)
            {
                zero(child.get());
            }
        };

        zero(this);
    }

    [[nodiscard]] std::string repr() const
    {
        return std::format("Value(data={:.4f}, grad={:.4f}{})", data, grad, label.empty() ? "" : std::format(", label='{}'", label));
    }
};
