#ifndef DUNE_ULTRAWEAK_DARCY_VELOCITY_UTILITY_HH
#define DUNE_ULTRAWEAK_DARCY_VELOCITY_UTILITY_HH

#include <dune/pdelab.hh>

template<typename DGF>
void writeVelocityFieldVTK(const DGF& dgf, const std::string filename="velocityField",
                           const std::size_t subsampling=1) {
  // set up vtk writer
  using GV = typename DGF::GridViewType;
  using VTKWriter = Dune::SubsamplingVTKWriter<GV>;
  const Dune::RefinementIntervals ref(subsampling);
  VTKWriter vtkwriter(dgf.getGridView(),ref);
  std::string vtkfile(filename);

  using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;
  vtkwriter.addCellData(std::make_shared<VTKF>(dgf, "velocity_field"));
  vtkwriter.write(vtkfile, Dune::VTK::appendedraw);
}

template<typename GV, typename F>
void writeAnalyticalVelocityFieldVTK(const GV& gv, const F& f, const std::size_t subsampling=1) {
  const auto dgf = Dune::PDELab::makeGridFunctionFromCallable(gv,f);
  writeVelocityFieldVTK(dgf, subsampling);
}

#endif  // DUNE_ULTRAWEAK_DARCY_VELOCITY_UTILITY_HH
