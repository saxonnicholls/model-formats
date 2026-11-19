# model-formats - the command-line tools: mf-inspect, mf-convert, mf-download. Each has a smoke run as a CTest
# (label tool) over files the run makes itself; mf-download's runs against localhost only in the unit tests.

add_executable(mf-inspect mf_inspect.cpp)
target_link_libraries(mf-inspect PRIVATE snicholls::formats_io)
add_executable(mf-convert mf_convert.cpp)
target_link_libraries(mf-convert PRIVATE snicholls::formats_io)
foreach(_tool mf-inspect mf-convert)
    model_formats_compile_options(${_tool})
endforeach()
if(MODEL_FORMATS_WITH_HUB)
    add_executable(mf-download mf_download.cpp)
    target_link_libraries(mf-download PRIVATE snicholls::formats_hub)
    model_formats_compile_options(mf-download)
endif()

if(MODEL_FORMATS_BUILD_TESTS)
    add_executable(mf_smoke_fixture smoke_fixture.cpp)
    target_link_libraries(mf_smoke_fixture PRIVATE snicholls::formats_io)
    model_formats_compile_options(mf_smoke_fixture)
    add_test(NAME tool.smoke COMMAND ${CMAKE_COMMAND} -DINSPECT=$<TARGET_FILE:mf-inspect> -DCONVERT=$<TARGET_FILE:mf-convert>
             -DFIXTURE=$<TARGET_FILE:mf_smoke_fixture> -DWORK=${CMAKE_CURRENT_BINARY_DIR}/smoke -P ${CMAKE_CURRENT_SOURCE_DIR}/smoke.cmake)
    set_tests_properties(tool.smoke PROPERTIES LABELS tool)
    if(MODEL_FORMATS_NETWORK_TESTS AND MODEL_FORMATS_WITH_HUB)
        add_test(NAME network.mf-download COMMAND mf-download hf-internal-testing/tiny-random-gpt2 --include "*.json" --cache
                 ${CMAKE_CURRENT_BINARY_DIR}/network-cache --max-bytes 209715200)
        set_tests_properties(network.mf-download PROPERTIES LABELS network TIMEOUT 600)
    endif()
endif()
