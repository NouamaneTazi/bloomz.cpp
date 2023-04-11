//
//  ContentView.swift
//  Bloomer
//
//  Created by Pedro Cuenca on 15/3/23.
//

import SwiftUI
import Dispatch
import bloomz

class ModelState: ObservableObject {
    @Published var model: OpaquePointer? = nil
    
    var modelPath: String {
        Bundle.main.path(forResource: "ggml-model-bloomz-560m-f16", ofType: "bin")!
    }

    func load() {
        DispatchQueue.global(qos: .userInitiated).async {
            let begin = Date()
            let model = load_model(self.modelPath)
            DispatchQueue.main.async { self.model = model }
            print("Loaded \(String(describing: model)) in \(Date().timeIntervalSince(begin))")
        }
    }
}

struct ContentView: View {
    @State private var prompt: String = "Translate \"Hi, how are you?\" into Spanish:\n"
    @State private var generated: String = ""
    @State private var generating: Bool = false
        
    @StateObject private var modelState = ModelState()
    private var modelIsLoaded: Bool { modelState.model != nil }
        
    func complete(from text: String) {
        guard let model = modelState.model else { return }
        
        generating.toggle()
        generated = ""
        
        DispatchQueue.global(qos: .userInteractive).async {
            guard let result = generate(model, text) else { print("Error"); return }
            DispatchQueue.main.async {
                generated = String(cString: result)
                generating = false
            }
        }
    }

    var body: some View {
        VStack {
            Image("bloom").resizable().aspectRatio(contentMode: .fit)
            HStack {
                TextField("Prompt", text: $prompt, axis: .vertical).lineLimit(2...5)
                    .textFieldStyle(.roundedBorder)
                Button("Complete") {
                    complete(from: prompt)
                }.buttonStyle(.borderedProminent).disabled(!modelIsLoaded)
            }
            if generating {
                ProgressView()
            }
            Text(generated)
            Spacer()
        }
        .padding()
        .onAppear {
            modelState.load()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
