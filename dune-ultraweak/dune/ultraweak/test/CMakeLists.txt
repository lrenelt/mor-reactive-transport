#convergence tests
add_executable("runConvergenceTest" runConvergenceTest.cc)
add_executable("runAllConvergenceTests" runAllConvergenceTests.cc)

# normal equation solutions
add_executable("solveNormalEq" solveNormalEq.cc)
add_executable("solve_transport_normal_eq_separated" solve_transport_normal_eq_separated.cc)

# reference solutions
add_executable("solve_darcy_mixed" solve_darcy_mixed.cc)
add_executable("solveDarcySIPG" solveDarcySIPG.cc)
add_executable("solve_transport_dg" solve_transport_dg.cc)

dune_symlink_to_source_files(FILES debug_parameters.ini)
dune_symlink_to_source_files(FILES conv_test_parameters.ini)
dune_symlink_to_source_files(FILES solver_config.ini)
