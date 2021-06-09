set(BINARY ${PROJECT_NAME}_test)

add_executable(${BINARY}
    ${PROJECT_SOURCE_DIR}/src/cudmd.cu
    cudmd.test.cu
)
target_include_directories(${BINARY} PRIVATE
    "${PROJECT_SOURCE_DIR}/lib/googletest/googlemock/include"
    "${PROJECT_SOURCE_DIR}/lib/googletest/googletest/include"
)
target_link_libraries(${BINARY} PRIVATE gmock_main gtest_main)
