// for (const LinearQuad::Element &element : elements)
//     {
//         percent = static_cast<unsigned int>(100.0 * elementsCreated / elementsTotal);
//         if (percent > lastPercent)
//         { // чтобы не выводить одно и то же значение много раз
//             std::cout << percent << "%\n";
//             lastPercent = percent;
//         }

//         for (int i = 0; i < 4; ++i)
//         {
//             coordMat(i, 0) = vertices[element.node_ids[i]].x;
//             coordMat(i, 1) = vertices[element.node_ids[i]].y;
//         }
//         Eigen::Matrix<double, 3, 2> coordMat1, coordMat2;
//         coordMat1.row(0) = coordMat.row(0);
//         coordMat1.row(1) = coordMat.row(1);
//         coordMat1.row(2) = coordMat.row(2);

//         coordMat2.row(0) = coordMat.row(0);
//         coordMat2.row(1) = coordMat.row(2);
//         coordMat2.row(2) = coordMat.row(3);
        
//         Ke = LinearQuad::element_stiffness(coordMat, D, 0.1);
//         // whether no enrichment or blend
//         if (heaviside_enriched_nodes[element.node_ids[0]] || heaviside_enriched_nodes[element.node_ids[1]] 
//             || heaviside_enriched_nodes[element.node_ids[2]] || heaviside_enriched_nodes[element.node_ids[3]]){
//             Ke_expanded.setZero();
//             for (int i = 0; i < 4; ++i) {
//                 for (int j = 0; j < 4; ++j) {
//                     Ke_expanded(4*i + 0, 4*j + 0) = Ke(2*i + 0, 2*j + 0);
//                     Ke_expanded(4*i + 0, 4*j + 1) = Ke(2*i + 0, 2*j + 1);
//                     Ke_expanded(4*i + 1, 4*j + 0) = Ke(2*i + 1, 2*j + 0);
//                     Ke_expanded(4*i + 1, 4*j + 1) = Ke(2*i + 1, 2*j + 1);
//                 }
//             }
//             // throw std::runtime_error("dsd");
//             // std::cout << "expanded\n" << Ke_expanded << std::endl;
//             FEMAssemble::addElementSparseUpperStiffness(element, Ke_expanded, triplets, node_offset, node_ndof, 4);
//         }else{
//             FEMAssemble::addElementSparseUpperStiffness(element, Ke, triplets, node_offset, node_ndof, 2);
//             // std::cout << "no enrichment\n" << Ke << std::endl;
//         }

//         elementsCreated++;
//     }