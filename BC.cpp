#include "BC.h"
// 
// void applyBC(double w, double h, int wn, int hn, Eigen::VectorXd& P, const std::vector<unsigned int>& node_offset, std::vector<int>& fixedDofs,
// std::vector<double>& fixedValues){
//     int off;
//     for (unsigned int i = 0; i < wn; i++)
//     {
//         // off = node_offset[i];
//         // fixedDofs.push_back(off);
//         // fixedDofs.push_back(off + 1);
//         // fixedValues.push_back(0);
//         // fixedValues.push_back(0);
//         // off = node_offset[(wn * (hn-1) + i)];
//         // if (i == 0){
//         //     off = node_offset[(wn * (hn-1) + i)];
//         //     fixedDofs.push_back(off);
//         //     fixedValues.push_back(-0.05);
//         //     off = node_offset[(wn * (hn-1) + i - wn)];
//         //     fixedDofs.push_back(off);
//         //     fixedValues.push_back(-0.05);
//         // }else if (i == wn - 1){
//         //     off = node_offset[(wn * (hn-1) + i)];
//         //     fixedDofs.push_back(off);
//         //     fixedValues.push_back(0.05);
//         //     off = node_offset[(wn * (hn-1) + i - wn)];
//         //     fixedDofs.push_back(off);
//         //     fixedValues.push_back(0.05);
//         // }
//         // fixedDofs.push_back(off);
//         // fixedDofs.push_back(off + 1);
//         // fixedValues.push_back(0.05);
//         // fixedValues.push_back(0.05);
//     }
//     // for (unsigned int i = hn * 0.2; i < hn * 0.8; i++){
//     //     P((wn*(i)+0)*2) = -10000000;
//     //     P((wn*(i)+wn-1)*2) = 10000000;
//     // }
//     // fixedDofs.push_back(node_offset[1]);
//     // fixedValues.push_back(0.0);
//     // fixedDofs.push_back(node_offset[1]+1);
//     // fixedValues.push_back(0.0);

//     // fixedDofs.push_back(node_offset[2]);
//     // fixedValues.push_back(0.0);

//     const double p = 10000000 * 0.1;
//     const double f_per_node = p*(w/(wn-1));
//     for (unsigned int i = 1; i < wn-1; i++){
//         off = node_offset[(wn * (hn-1) + i)];
//         P(off+1) = f_per_node*2;
//         off = node_offset[i];
//         P(off+1) = -f_per_node*2;
//     }
//     off = node_offset[(wn * (hn-1))];
//     P(off+1) = f_per_node;
//     off = node_offset[(wn * (hn-1))+wn-1];
//     P(off+1) = f_per_node;

//     off = node_offset[0];
//     P(off+1) = -f_per_node;
//     off = node_offset[wn-1];
//     P(off+1) = -f_per_node;
// }


// void applyBC(
//     double w,
//     double h,
//     int wn,
//     int hn,
//     double thickness,
//     Eigen::VectorXd& P,
//     const std::vector<unsigned int>& node_offset,
//     std::vector<int>& fixedDofs,
//     std::vector<double>& fixedValues
// ) {
//     int off;

//     // Murakami 1.3:
//     // strip with a single edge transverse crack under uniaxial tension
//     //
//     // Crack should be defined separately in crack.txt, for example:
//     // (0, h / 2) -> (a, h / 2)
//     //
//     // Here:
//     // W = w = 1.0
//     // a = 0.5
//     // thickness = 0.1
//     // sigma = 10000

//     const double sigma = 10000.0;

//     // Uniform traction on top and bottom edges.
//     // The boundary is divided into segments of length dx.
//     const double dx = w / static_cast<double>(wn - 1);

//     // Force contribution for one boundary segment.
//     const double f_segment = sigma * thickness * dx;

//     // Top edge: +Y
//     // Bottom edge: -Y
//     //
//     // Corner nodes receive half contribution.
//     // Interior edge nodes receive full contribution.
//     for (int i = 0; i < wn; ++i) {
//         const double weight = (i == 0 || i == wn - 1) ? 0.5 : 1.0;

//         // Top edge node: y = h
//         off = node_offset[wn * (hn - 1) + i];
//         P(off + 1) += weight * f_segment;

//         // Bottom edge node: y = 0
//         off = node_offset[i];
//         P(off + 1) -= weight * f_segment;
//     }

//     // Minimal reference constraints.
//     // They are not meant to model a clamped support.
//     // They only remove rigid body modes:
//     // 1) translation in X
//     // 2) translation in Y
//     // 3) rigid rotation

//     // Bottom-left node: fix ux and uy
//     off = node_offset[0];

//     fixedDofs.push_back(off);
//     fixedValues.push_back(0.0);

//     fixedDofs.push_back(off + 1);
//     fixedValues.push_back(0.0);

//     // Top-left node: fix ux only to remove rigid rotation
//     off = node_offset[wn * (hn - 1)];

//     fixedDofs.push_back(off);
//     fixedValues.push_back(0.0);
// }

void applyBC(
    double w,
    double h,
    int wn,
    int hn,
    double thickness,
    Eigen::VectorXd& P,
    const std::vector<unsigned int>& node_offset,
    std::vector<int>& fixedDofs,
    std::vector<double>& fixedValues
) {
    const double sigma = 10000.0;
    const double dx = w / static_cast<double>(wn - 1);
    const double f_segment = sigma * thickness * dx;


    int off;

    // ------------------------------------------------------------
    // Top edge load only: equivalent to ANSYS SFL,5,PRES,-sigma
    // ------------------------------------------------------------
    for (int i = 0; i < wn; ++i) {
        const double weight = (i == 0 || i == wn - 1) ? 0.5 : 1.0;

        // top edge
        off = node_offset[wn * (hn - 1) + i];
        P(off + 1) += weight * f_segment;

        // bottom edge
        off = node_offset[i];
        P(off + 1) -= weight * f_segment;
    }   

    // bottom-left: UX = 0, UY = 0
    off = node_offset[0];
    fixedDofs.push_back(off);
    fixedValues.push_back(0.0);

    fixedDofs.push_back(off + 1);
    fixedValues.push_back(0.0);

    // top-left: UX = 0
    off = node_offset[wn * (hn - 1)];
    fixedDofs.push_back(off);
    fixedValues.push_back(0.0);
}