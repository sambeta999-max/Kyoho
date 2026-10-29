# SM86 (Ampere) shader tools

Tools behind `src/shader_recompiler/frontend/ampere` and its tests. They need the CUDA Toolkit
(`nvdisasm`, `nvcc`); the hardware tools also need an SM 8.x GPU. Generated files are committed, so
none of this is required to build or test Eden.

## Opcode table

`ampere.inc` and `ampere_decode_vectors.inc` are generated from `sm86_opcodes.json`, a sweep of the
12-bit opcode space through `nvdisasm -b SM86` with one valid encoding per opcode:

```sh
python3 gen_ampere_inc.py sm86_opcodes.json \
    ../../src/shader_recompiler/frontend/ampere/ampere.inc \
    ../../src/tests/shader_recompiler/ampere_decode_vectors.inc
```

To redo the sweep (about an hour, one `nvdisasm` call per probe):

```sh
python3 sweep_opcodes.py <workdir> sm86_opcodes.json   # random fills of every opcode
python3 sweep_forms.py <workdir> sm86_opcodes.json     # missing forms of known bases
```

`bitprobe.py` flips single bits of an encoding and prints how the disassembly changes; field
positions in the frontend were checked with it.

## Engine class changes

`class_diff.py` compares two NVIDIA class headers (open-gpu-doc) and annotates each changed
method with the Eden register field it lands in. The Maxwell to Ampere reports and a summary of
what Eden has to read differently are in [`reports/`](reports/README.md).

## Compiler output vectors

`ampere_decode_compiler_vectors.inc` holds real `nvcc -arch=sm_86` output:

```sh
nvcc -arch=sm_86 -cubin -O3 -o corpus.cubin templates/corpus.cu
nvcc -arch=sm_86 -cubin -O0 -G -o corpus_g.cubin templates/corpus.cu
python3 gen_compiler_vectors.py corpus.cubin corpus_g.cubin \
    > ../../src/tests/shader_recompiler/ampere_decode_compiler_vectors.inc
```

## Hardware vectors

`ampere_hardware_vectors.inc` holds results of exact encodings executed on a GA102. The templates
in `templates/` are compiled so that each test instruction has a known register assignment; the
instruction is then patched to the encoding under test and run with the CUDA driver API.
Variable-latency instructions keep the scoreboard bits of a compiler-generated template.

```sh
mkdir hw && cd hw
nvcc -O2 -o oracle ../oracle.cpp -lcuda
nvcc -arch=sm_86 -cubin -O3 -o t_ops.cubin ../templates/t_ops.cu
nvcc -arch=sm_86 -cubin -O3 -o t_f2i.cubin ../templates/t_f2i.cu
python3 ../gen_hw_vectors.py . ../hwrun.py ../../../src/tests/shader_recompiler/ampere_hardware_vectors.inc
```

Template encodings are matched byte for byte; if a different `nvcc` schedules them differently,
update the template constants at the top of `gen_hw_vectors.py`.
