//
//  ContentView.swift
//  Bloomer
//
//  Created by Pedro Cuenca on 15/3/23.
//

import SwiftUI
import Dispatch
import bloomz

struct ContentView: View {
    @State private var prompt: String = "This is not"
    @State private var generated: String = ""
    @State private var generating: Bool = false
    
    var modelPath: String {
        Bundle.main.path(forResource: "ggml-model-bloomz-560m-f16", ofType: "bin")!
    }
    
    func complete(from text: String) {
        generating.toggle()
        generated = ""
        
        DispatchQueue.global(qos: .userInteractive).async {
            guard let result = generate(modelPath, text) else { print("Error"); return }
            DispatchQueue.main.async {
                generated = String(cString: result)
                generating = false
            }
        }
    }

    var body: some View {
        VStack {
            Image("bloom").resizable().aspectRatio(contentMode: .fit)
            TextField("Prompt", text: $prompt).lineLimit(5)
                .textFieldStyle(.roundedBorder)
                .onSubmit {
                        complete(from: prompt)
                }
            if generating {
                ProgressView()
            }
            Text(generated)
            Spacer()
        }
        .padding()
        .onAppear {
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
