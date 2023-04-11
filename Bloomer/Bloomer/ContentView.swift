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
    @State private var status: String = ""
        
    @StateObject private var modelState = ModelState()
    private var modelIsLoaded: Bool { modelState.model != nil }
        
    func listenToNotifications() {
        NotificationCenter.default.addObserver(forName: NSNotification.Name("decoded.token.received"), object: nil, queue: nil) { decoded in
            generated = generated.appending(decoded.object as! String)
        }
    }
    
    func complete(from text: String) {
        guard let model = modelState.model else { return }
        
        generating.toggle()
        generated = text
        status = ""
        
        DispatchQueue.global(qos: .userInteractive).async {
            func token_callback(char_ptr: UnsafePointer<CChar>?) {
                guard let char_ptr = char_ptr else { return }
                
                // We can't pass a "function that captures context" as a C function pointer, so we resort to posting a notification
                DispatchQueue.main.async {
                    let str = String(cString: char_ptr)
                    NotificationCenter.default.post(name: NSNotification.Name("decoded.token.received"), object: str)
                }
            }
            
            let msPerToken = generate(model, text, token_callback)
            DispatchQueue.main.async {
                status = String(format: "%.2f ms/token", msPerToken)
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
            Text(generated).frame(maxWidth: .infinity, alignment: .leading)//.multilineTextAlignment(.leading)
            if generating {
                ProgressView().padding()
            }
            Spacer()
            if status != "" {
                Text(status).font(.system(size: 14)).padding().frame(maxWidth: .infinity).background(Color(white: 0.9))
            }
        }
        .padding()
        .onAppear {
            listenToNotifications()
            modelState.load()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
