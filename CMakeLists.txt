cmake_minimum_required(VERSION 3.10)
project(ufldLaneDetection)

set(CMAKE_CXX_STANDARD 14)

# 1. Configuration des flags de compilation
add_compile_definitions(
    FLATBUFFERS_VERSION_MAJOR=1
    FLATBUFFERS_VERSION_MINOR=14
)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)

# 2. Trouver les dépendances principales
find_package(OpenCV REQUIRED)

# Configuration FlatBuffers
set(FLATBUFFERS_INCLUDE_DIR "/usr/local/include")
set(FLATBUFFERS_LIBRARY "/usr/local/lib/libflatbuffers.a")

# 3. Configuration TensorFlow Lite - Version robuste
set(TENSORFLOW_LITE_INCLUDE_DIR "/usr/local/include/tensorflow/lite")
set(TENSORFLOW_LITE_LIBRARY "/usr/local/lib/libtensorflowlite.so")

# Vérification alternative des headers
if(NOT EXISTS "${TENSORFLOW_LITE_INCLUDE_DIR}/interpreter.h")
    message(WARNING "Header interpreter.h manquant - tentative de recherche alternative")
    
    # Essai avec les headers des delegates GPU si disponibles
    if(EXISTS "${TENSORFLOW_LITE_INCLUDE_DIR}/delegates/gpu/api.h")
        message(STATUS "Utilisation des headers GPU delegates")
        list(APPEND TENSORFLOW_LITE_INCLUDE_DIR "${TENSORFLOW_LITE_INCLUDE_DIR}/delegates/gpu")
    else()
        message(FATAL_ERROR "Aucun header TensorFlow Lite trouvé")
    endif()
endif()

# Configuration de la bibliothèque
add_library(tensorflow_lite SHARED IMPORTED)
set_target_properties(tensorflow_lite PROPERTIES
    IMPORTED_LOCATION "${TENSORFLOW_LITE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_LITE_INCLUDE_DIR};${FLATBUFFERS_INCLUDE_DIR}"
)

# Ajoutez ceci avant les include_directories
find_path(FLATBUFFERS_INCLUDE_DIR "flatbuffers/flatbuffers.h"
  PATHS "/usr/local/include" "/usr/include"
  REQUIRED
)

# Modifiez les include_directories comme suit :
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORFLOW_LITE_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}/..  # Pour les includes imbriqués
)

# Version vidéo
add_executable(ufld_video src/main.cpp src/UltrafastLaneDetector.cpp)
target_link_libraries(ufld_video PRIVATE 
    ${OpenCV_LIBS} 
    tensorflow_lite
    ${FLATBUFFERS_LIBRARY}
    pthread dl
)

# Version images
add_executable(ufld_images src/main_images.cpp src/UltrafastLaneDetector.cpp)
target_link_libraries(ufld_images PRIVATE 
    ${OpenCV_LIBS} 
    tensorflow_lite
    ${FLATBUFFERS_LIBRARY}
    pthread dl
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)