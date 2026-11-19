# model-formats - the tools' smoke run (CTest tool.smoke): every conversion path once, each output inspected.
#   cmake -DINSPECT=... -DCONVERT=... -DFIXTURE=... -DWORK=<dir> -P smoke.cmake

file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    message(STATUS "${ARGN}\n${out}${err}")
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "failed (${rc}): ${ARGN}")
    endif()
endfunction()
run(${FIXTURE} ${WORK})
run(${INSPECT} ${WORK}/tiny_gpt2.bin)
run(${CONVERT} ${WORK}/tiny_gpt2.bin ${WORK}/hf)                                   # llm.c -> Hugging Face directory
file(COPY ${WORK}/tokenizer.json DESTINATION ${WORK}/hf)
run(${INSPECT} ${WORK}/hf)
run(${CONVERT} ${WORK}/hf ${WORK}/from_hf.gguf --name tiny)                         # Hugging Face -> GGUF
run(${INSPECT} ${WORK}/from_hf.gguf)
run(${CONVERT} ${WORK}/tiny_gpt2.bin ${WORK}/direct.gguf --tokenizer ${WORK}/tokenizer.json --name tiny)   # llm.c -> GGUF
run(${INSPECT} --no-tensors ${WORK}/direct.gguf)
run(${CONVERT} ${WORK}/from_hf.gguf ${WORK}/tensors.npz)                            # tensor by tensor
run(${CONVERT} ${WORK}/tensors.npz ${WORK}/tensors.safetensors --dtype bf16)
run(${INSPECT} ${WORK}/tensors.safetensors)
execute_process(COMMAND ${INSPECT} ${WORK}/tokenizer.json RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
if(rc EQUAL 0)
    message(FATAL_ERROR "mf-inspect accepted a JSON file as a model")
endif()
