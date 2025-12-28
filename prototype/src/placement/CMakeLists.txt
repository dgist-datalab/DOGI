
add_library(placement dogi.cc)


target_include_directories(placement PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(placement PRIVATE cxx_std_17)
target_compile_options(placement PRIVATE -march=znver3)
#target_compile_options(placement PRIVATE -O2 -march=znver3)
#target_compile_options(placement PRIVATE -march=native)
#target_compile_options(placement PRIVATE -O3 -march=native)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
