set(GAME_SOURCES
  bridge.cc
  bridge.h
  bridge/bridge_scoring.cc
  bridge/bridge_scoring.h
  bridge_uncontested_bidding.cc
  bridge_uncontested_bidding.h
  bridgev2.cc
  bridgev2.h
  bridge/bridgev2_scoring.cc
  bridge/bridgev2_scoring.h
  bridgev2_uncontested_bidding.cc
  bridgev2_uncontested_bidding.h
)

add_library (games OBJECT ${GAME_SOURCES})

target_include_directories (games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


add_library(bridge_double_dummy_solver OBJECT
  bridge/double_dummy_solver/include/dll.h
  bridge/double_dummy_solver/include/portab.h
  bridge/double_dummy_solver/src/ABsearch.cpp
  bridge/double_dummy_solver/src/ABsearch.h
  bridge/double_dummy_solver/src/ABstats.cpp
  bridge/double_dummy_solver/src/ABstats.h
  bridge/double_dummy_solver/src/CalcTables.cpp
  bridge/double_dummy_solver/src/CalcTables.h
  bridge/double_dummy_solver/src/dds.cpp
  bridge/double_dummy_solver/src/dds.h
  bridge/double_dummy_solver/src/DealerPar.cpp
  bridge/double_dummy_solver/src/debug.h
  bridge/double_dummy_solver/src/dump.cpp
  bridge/double_dummy_solver/src/dump.h
  bridge/double_dummy_solver/src/File.cpp
  bridge/double_dummy_solver/src/File.h
  bridge/double_dummy_solver/src/Init.cpp
  bridge/double_dummy_solver/src/Init.h
  bridge/double_dummy_solver/src/LaterTricks.cpp
  bridge/double_dummy_solver/src/LaterTricks.h
  bridge/double_dummy_solver/src/Memory.cpp
  bridge/double_dummy_solver/src/Memory.h
  bridge/double_dummy_solver/src/Moves.cpp
  bridge/double_dummy_solver/src/Moves.h
  bridge/double_dummy_solver/src/Par.cpp
  bridge/double_dummy_solver/src/parallel.h
  bridge/double_dummy_solver/src/PBN.cpp
  bridge/double_dummy_solver/src/PBN.h
  bridge/double_dummy_solver/src/PlayAnalyser.cpp
  bridge/double_dummy_solver/src/PlayAnalyser.h
  bridge/double_dummy_solver/src/QuickTricks.cpp
  bridge/double_dummy_solver/src/QuickTricks.h
  bridge/double_dummy_solver/src/Scheduler.cpp
  bridge/double_dummy_solver/src/Scheduler.h
  bridge/double_dummy_solver/src/SolveBoard.cpp
  bridge/double_dummy_solver/src/SolveBoard.h
  bridge/double_dummy_solver/src/SolverIF.cpp
  bridge/double_dummy_solver/src/SolverIF.h
  bridge/double_dummy_solver/src/System.cpp
  bridge/double_dummy_solver/src/System.h
  bridge/double_dummy_solver/src/ThreadMgr.cpp
  bridge/double_dummy_solver/src/ThreadMgr.h
  bridge/double_dummy_solver/src/Timer.cpp
  bridge/double_dummy_solver/src/Timer.h
  bridge/double_dummy_solver/src/TimerGroup.cpp
  bridge/double_dummy_solver/src/TimerGroup.h
  bridge/double_dummy_solver/src/TimerList.cpp
  bridge/double_dummy_solver/src/TimerList.h
  bridge/double_dummy_solver/src/TimeStat.cpp
  bridge/double_dummy_solver/src/TimeStat.h
  bridge/double_dummy_solver/src/TimeStatList.cpp
  bridge/double_dummy_solver/src/TimeStatList.h
  bridge/double_dummy_solver/src/TransTable.h
  bridge/double_dummy_solver/src/TransTableL.cpp
  bridge/double_dummy_solver/src/TransTableL.h
  bridge/double_dummy_solver/src/TransTableS.cpp
  bridge/double_dummy_solver/src/TransTableS.h
)
target_include_directories (bridge_double_dummy_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bridge_double_dummy_solver PUBLIC DDS_NO_STATIC_INIT)

add_executable(bridge_test bridge_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(bridge_test bridge_test)

add_executable(bridgev2_test bridgev2_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(bridgev2_test bridgev2_test)


add_executable(y_test y_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(y_test y_test)

