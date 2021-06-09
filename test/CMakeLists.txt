set(TARGET ${PROJECT_NAME}_test)

add_executable(${TARGET}
    ${PROJECT_SOURCE_DIR}/src/cudmd.cu
    cudmd.test.cu
)
target_include_directories(${TARGET} PRIVATE
    "${PROJECT_SOURCE_DIR}/lib/googletest/googlemock/include"
    "${PROJECT_SOURCE_DIR}/lib/googletest/googletest/include"
    "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(${TARGET} PRIVATE gmock_main gtest_main)
