// lattice_viz.cpp
//
// Standalone test/visualization tool for the neighbor-list generators
// (neighbors_2d_nn, neighbors_2d_nnn, neighbors_triangular).
//
// It renders each lattice to an SVG file: nodes as circles, bonds as lines.
// Periodic-boundary ("wrap-around") bonds are drawn as short dashed stubs
// pointing off the edge of the lattice (instead of a line across the whole
// system) so the picture stays readable while still letting you check that
// the wrap-around is going to the right place.
//
// Build:   g++ -O2 -std=c++17 lattice_viz.cpp -o lattice_viz
// Run:     ./lattice_viz
// Output:  nn_L8.svg, nnn_L8.svg, triangular_L8.svg  (open in a browser)

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cassert>
#include <iostream>
#include <filesystem>

#include "core/lattice/lattice.hpp"

// ---------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------
//
// All the functionality of this file is made by Claude Sonnet 5 Medium (2026)
// 
// geometry = "square"      -> node (i,j) at (x=j, y=i)
// geometry = "triangular"  -> node (i,j) at (x=j+0.5*(i%2), y=i*sqrt(3)/2)
//
// The bond from `id` to `neigh[id*k+n]` is drawn as a straight line if the
// neighbor lies within one lattice step in the *unwrapped* plane; otherwise
// it is a periodic wrap and is drawn as a short dashed stub in the
// direction of that neighbor, clipped at the lattice boundary.

struct Point { double x, y; };

static Point node_pos(int i, int j, const std::string& geometry) {
    if (geometry == "triangular") {
        return { j + 0.5 * (i % 2), i * std::sqrt(3.0) / 2.0 };
    }
    return { (double)j, (double)i };
}

void visualize_lattice_svg(const std::string& filename,
                            int Lx, int Ly,
                            const std::vector<int>& neigh,
                            int n_neighbors,
                            const std::string& geometry = "square",
                            const std::string& title = "") {
    int N = Lx * Ly;
    assert((int)neigh.size() == N * n_neighbors);

    const double scale = 60.0;   // px per lattice spacing
    const double margin = 60.0;
    const double r = 7.0;        // node radius

    // bounding box in lattice units
    double max_x = 0, max_y = 0;
    std::vector<Point> pos(N);
    for (int i = 0; i < Ly; i++) {
        for (int j = 0; j < Lx; j++) {
            Point p = node_pos(i, j, geometry);
            pos[i * Lx + j] = p;
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }
    }

    double W = max_x * scale + 2 * margin;
    double H = max_y * scale + 2 * margin;

    auto X = [&](double x) { return margin + x * scale; };
    auto Y = [&](double y) { return margin + y * scale; };

    std::ofstream f(filename);
    f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W
      << "\" height=\"" << (H + (title.empty() ? 0 : 30)) << "\" "
      << "viewBox=\"0 0 " << W << " " << (H + (title.empty() ? 0 : 30)) << "\">\n";
    f << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

    if (!title.empty()) {
        f << "<text x=\"" << W / 2 << "\" y=\"20\" font-family=\"sans-serif\" "
          << "font-size=\"16\" text-anchor=\"middle\">" << title << "</text>\n";
    }
    double y_off = title.empty() ? 0 : 30;

    // Draw bonds first (so nodes sit on top). Each undirected bond is stored
    // twice (once from each endpoint) - only draw it once, when id < nbr.
    for (int i = 0; i < Ly; i++) {
        for (int j = 0; j < Lx; j++) {
            int id = i * Lx + j;
            Point p0 = pos[id];

            for (int n = 0; n < n_neighbors; n++) {
                int nbr = neigh[id * n_neighbors + n];
                if (nbr < id) continue; // avoid drawing every bond twice

                int ni = nbr / Lx, nj = nbr % Lx;
                Point p1 = pos[nbr];

                // Is this a periodic wrap, or a genuine local bond?
                // A wrap happens whenever the raw (unwrapped) index distance
                // in row or column is more than 1 step away in either
                // direction once you account for the actual small offsets
                // used above (|di|<=1 and |dj|<=1 in *unwrapped* space).
                int di = ni - i, dj = nj - j;
                bool wrap_i = std::abs(di) > 1 && std::abs(di) != Ly - 1;
                bool wrap_j = std::abs(dj) > 1 && std::abs(dj) != Lx - 1;
                bool is_wrap = (std::abs(di) == Ly - 1 && Ly > 2) ||
                               (std::abs(dj) == Lx - 1 && Lx > 2) ||
                               wrap_i || wrap_j;

                if (!is_wrap) {
                    f << "<line x1=\"" << X(p0.x) << "\" y1=\"" << Y(p0.y) + y_off
                      << "\" x2=\"" << X(p1.x) << "\" y2=\"" << Y(p1.y) + y_off
                      << "\" stroke=\"#4472c4\" stroke-width=\"1.5\"/>\n";
                } else {
                    // draw a short dashed stub from id toward nbr's direction,
                    // clipped near id, to indicate a periodic-boundary bond
                    // (direction found via minimum-image convention)
                    double ddx = nj - j, ddy = ni - i;
                    
                    if (ddx > Lx / 2.0) ddx -= Lx;
                    if (ddx < -Lx / 2.0) ddx += Lx;

                    if (ddy > Ly / 2.0) ddy -= Ly;
                    if (ddy < -Ly / 2.0) ddy += Ly;

                    double len = std::sqrt(ddx * ddx + ddy * ddy);
                    if (len < 1e-9) len = 1;
                    double ux = ddx / len, uy = ddy / len;
                    double stub = 0.35; // lattice units
                    f << "<line x1=\"" << X(p0.x) << "\" y1=\"" << Y(p0.y) + y_off
                      << "\" x2=\"" << X(p0.x + ux * stub) << "\" y2=\"" << Y(p0.y + uy * stub) + y_off
                      << "\" stroke=\"#c00000\" stroke-width=\"1.5\" stroke-dasharray=\"3,3\"/>\n";
                    double uux = -ux, uuy = -uy;
                    f << "<line x1=\"" << X(p1.x) << "\" y1=\"" << Y(p1.y) + y_off
                      << "\" x2=\"" << X(p1.x + uux * stub) << "\" y2=\"" << Y(p1.y + uuy * stub) + y_off
                      << "\" stroke=\"#c00000\" stroke-width=\"1.5\" stroke-dasharray=\"3,3\"/>\n";
                }
            }
        }
    }

    // Draw nodes
    for (int i = 0; i < Ly; i++) {
        for (int j = 0; j < Lx; j++) {
            Point p = pos[i * Lx + j];
            f << "<circle cx=\"" << X(p.x) << "\" cy=\"" << Y(p.y) + y_off
              << "\" r=\"" << r << "\" fill=\"#1f1f1f\"/>\n";
        }
    }

    f << "</svg>\n";
    f.close();
    std::cout << "wrote " << filename << " (" << N << " nodes, "
              << n_neighbors << " neighbors/node, geometry=" << geometry << ")\n";
}

// ---------------------------------------------------------------------
// Test main
// ---------------------------------------------------------------------

int main() {
    int L = 8;

    std::filesystem::create_directories("temp/LatticeViz");

    auto nn = neighbors_2d_nn(L);
    visualize_lattice_svg("temp/LatticeViz/nn_L8.svg", L, L, nn, 4, "square",
                           "2D nearest-neighbor lattice (L=8)");

    auto nnn = neighbors_2d_nnn(L);
    visualize_lattice_svg("temp/LatticeViz/nnn_L8.svg", L, L, nnn, 4, "square",
                           "2D next-nearest-neighbor (diagonal) lattice (L=8)");

    int Lx = 8, Ly = 8;
    auto tri = neighbors_triangular(Lx, Ly);
    visualize_lattice_svg("temp/LatticeViz/triangular_L8.svg", Lx, Ly, tri, 6, "triangular",
                           "Triangular lattice (Lx=Ly=8)");

    auto honey = neighbors_honeycomb(Lx, Ly);
    visualize_lattice_svg("temp/LatticeViz/honeycomb_L8.svg", Lx, Ly, honey, 3, "honeycomb",
                           "Honeycomb lattice (Lx=Ly=8)");

    return 0;
}