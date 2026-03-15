#pragma once

#include "value.h"
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

struct TraceResult
{
    std::set<Value*> nodes;
    std::set<std::pair<Value*, Value*>> edges;
};

inline TraceResult trace(ValuePtr root)
{
    TraceResult result;

    std::function<void(Value*)> build = [&](Value* v)
    {
        if (result.nodes.contains(v)) return;
        result.nodes.insert(v);
        for (auto& child : v->_prev)
        {
            result.edges.insert({child.get(), v});
            build(child.get());
        }
    };

    build(root.get());
    return result;
}

inline void draw_dot(ValuePtr root, std::string_view base_name = "graph", bool render_svg = true)
{
    auto [nodes, edges] = trace(root);

    std::ostringstream dot;
    dot << "digraph {\n";
    dot << "    rankdir=LR;\n";
    dot << "    node [fontname=\"Helvetica\"];\n\n";

    for (Value* n : nodes)
    {
        std::string uid = std::to_string(reinterpret_cast<std::uintptr_t>(n));
        dot << std::format(
                "    \"{}\" [shape=record, label=\"{{ {} | data {:.4f} | grad {:.4f} }}\"];\n",
                uid, n->label, n->data, n->grad);

        if (!n->_op.empty())
        {
            std::string op_uid = uid + n->_op;
            dot << std::format("    \"{}\" [label=\"{}\"];\n", op_uid, n->_op);
            dot << std::format("    \"{}\" -> \"{}\";\n", op_uid, uid);
        }
    }

    dot << "\n";

    for (auto& [child, parent] : edges)
    {
        std::string child_uid = std::to_string(reinterpret_cast<std::uintptr_t>(child));
        std::string par_uid = std::to_string(reinterpret_cast<std::uintptr_t>(parent));
        dot << std::format("    \"{}\" -> \"{}\";\n", child_uid, par_uid + parent->_op);
    }

    dot << "}\n";

    std::string dot_path = std::string(base_name) + ".dot";
    {
        std::ofstream f(dot_path);
        if (!f) throw std::runtime_error("Cannot open " + dot_path);
        f << dot.str();
    }

    if (render_svg)
    {
        std::string svg_path = std::string(base_name) + ".svg";
        std::string cmd = std::format("dot -Tsvg \"{}\" -o \"{}\"", dot_path, svg_path);
        int rc = std::system(cmd.c_str());
        if (rc != 0)
        {
            throw std::runtime_error("dot rendering failed (is graphviz installed?)");
        }
        std::cout << "[draw_dot] saved: " << svg_path << "\n";
    }
    else
    {
        std::cout << "[draw_dot] saved: " << dot_path << " (no SVG rendered)\n";
    }
}
