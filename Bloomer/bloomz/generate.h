//
//  generator.h
//  Bloomer
//
//  Created by Pedro Cuenca on 15/3/23.
//

#ifndef generator_h
#define generator_h

#ifdef __cplusplus
extern "C" {
#endif

// Opaque type defined in generate.cpp
typedef struct _model_context BloomModel;

extern const BloomModel * load_model(const char * model_path);
extern const char * generate(const BloomModel * model, const char * prompt);

#ifdef __cplusplus
}
#endif

#endif /* generator_h */
