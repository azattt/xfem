#include "BC.h"

void applyBC(double w, double h, int wn, int hn, Eigen::VectorXd& P, const std::vector<unsigned int>& node_offset, std::vector<int>& fixedDofs,
std::vector<double>& fixedValues){
    int off;
    for (unsigned int i = 0; i < wn; i++)
    {
        off = node_offset[i];
        fixedDofs.push_back(off);
        fixedDofs.push_back(off + 1);
        fixedValues.push_back(0);
        fixedValues.push_back(0);
        off = node_offset[(wn * (hn-1) + i)];
        // if (i == 0){
        //     off = node_offset[(wn * (hn-1) + i)];
        //     fixedDofs.push_back(off);
        //     fixedValues.push_back(-0.05);
        //     off = node_offset[(wn * (hn-1) + i - wn)];
        //     fixedDofs.push_back(off);
        //     fixedValues.push_back(-0.05);
        // }else if (i == wn - 1){
        //     off = node_offset[(wn * (hn-1) + i)];
        //     fixedDofs.push_back(off);
        //     fixedValues.push_back(0.05);
        //     off = node_offset[(wn * (hn-1) + i - wn)];
        //     fixedDofs.push_back(off);
        //     fixedValues.push_back(0.05);
        // }
        // fixedDofs.push_back(off);
        // fixedDofs.push_back(off + 1);
        // fixedValues.push_back(0.05);
        // fixedValues.push_back(0.05);
    }
    // for (unsigned int i = hn * 0.2; i < hn * 0.8; i++){
    //     P((wn*(i)+0)*2) = -10000000;
    //     P((wn*(i)+wn-1)*2) = 10000000;
    // }
    const double p = 100000000;
    const double f_per_node = p*(w/(wn-1));
    for (unsigned int i = 1; i < wn-1; i++){
        off = node_offset[(wn * (hn-1) + i)];
        P(off+1) = f_per_node*2;
    }
    off = node_offset[(wn * (hn-1))];
    P(off+1) = f_per_node;
    off = node_offset[(wn * (hn-1))+wn-1];
    P(off+1) = f_per_node;
}