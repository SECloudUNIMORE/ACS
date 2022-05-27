cmake_minimum_required(VERSION 3.14)
project(c_code C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-Wall -Wconversion -Wformat -O9")

add_executable(main_files_6 src_cli/main_files.c src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c)
add_executable(main_files_6_32 src_cli/main_files.c src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c)

add_executable(main_files_2 src_cli/main_files.c src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c)
add_executable(main_files_2_32 src_cli/main_files.c src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c)


add_executable(sparse_speed src_cli/sparse_speed.c src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c misc/timings.c misc/timings.h)
add_executable(sparse_speed_32 src_cli/sparse_speed.c src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c misc/timings.c misc/timings.h)


add_executable(main_embedded src_cli/main_embedded.c src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c)
add_executable(main_embedded_32 src_cli/main_embedded.c src/byod.c src/byod.h src/config.h misc/files.h src/types.h src/types.c)

add_executable(verify_bitmap_files src/byod.c src/byod.h src/config.h misc/files.c misc/files.h src/types.h src/types.c)

add_executable(verify_files src_cli/verify_files.c src/config.h misc/files.c misc/files.h src/types.h src/types.c)

# 32 bit targets options
target_link_options(main_files_2_32 PUBLIC -m32)
target_link_options(main_files_6_32 PUBLIC -m32)
target_compile_options(sparse_speed_32 PUBLIC -m32)
target_link_options(sparse_speed_32 PUBLIC -m32)
target_compile_options(main_embedded_32 PUBLIC -m32)
target_link_options(main_embedded_32 PUBLIC -m32)

target_link_libraries(sparse_speed m)
target_link_libraries(sparse_speed_32 m)

#target_compile_definitions(test_compressed_opt PUBLIC limit=1 compressed=1 use_tables=1)
target_compile_definitions(main_files_2 PUBLIC limit=1000000 compressed=1 use_tables=1 block_type_size=1 rle_bitsize=2)
target_compile_definitions(main_files_2_32 PUBLIC limit=1000000 compressed=1 use_tables=1 block_type_size=1 rle_bitsize=2)
target_compile_definitions(main_files_6 PUBLIC limit=1000000 compressed=1 use_tables=1 block_type_size=1 rle_bitsize=6)
target_compile_definitions(main_files_6_32 PUBLIC limit=1000000 compressed=1 use_tables=1 block_type_size=1 rle_bitsize=6)

target_compile_definitions(sparse_speed PUBLIC compressed=1 use_tables=1 block_type_size=1 rle_bitsize=2)
target_compile_definitions(sparse_speed_32 PUBLIC compressed=1 use_tables=1 block_type_size=1 rle_bitsize=2)

target_compile_definitions(verify_bitmap_files PUBLIC limit=1000000 block_type_size=1 rle_bitsize=6)

target_compile_definitions(main_embedded PUBLIC limit=1000000 compressed=1 use_tables=1 block_type_size=1 rle_bitsize=2)
target_compile_definitions(main_embedded_32 PUBLIC limit=1000000 compressed=1 use_tables=1 block_type_size=1 rle_bitsize=2)

target_compile_definitions(verify_files PUBLIC block_type_size=1 )
target_compile_definitions(verify_files PUBLIC block_type_size=1 )

#target_compile_definitions(test_plain_opt PUBLIC limit=1 use_tables=1 )
#target_compile_definitions(test_compressed PUBLIC limit=1 compressed=1 use_table=0 )
#target_compile_definitions(test_plain PUBLIC limit=1 use_tables=0 )