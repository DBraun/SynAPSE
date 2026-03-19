import { useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { getTwClasses } from '../theme';

export function InfoModal() {
  const { mode } = useTheme();
  const twClasses = getTwClasses(mode);
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className={`px-3 py-1 text-sm ${twClasses.buttonSecondary} rounded`}
      >
        What is this?
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className={`${twClasses.bgElevated} rounded-lg shadow-xl max-w-2xl mx-4 max-h-[90vh] overflow-y-auto`}>
            <div className={`px-6 py-4 border-b ${twClasses.borderSubtle} flex items-center justify-between`}>
              <h2 className={`text-xl font-semibold ${twClasses.textPrimary}`}>Can a computer match sounds to synth presets?</h2>
              <button
                onClick={() => setIsOpen(false)}
                className={twClasses.buttonGhost + ' text-2xl leading-none'}
              >
                &times;
              </button>
            </div>

            <div className={`px-6 py-4 space-y-4 ${twClasses.textSecondary}`}>
              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>The Goal</h3>
                <p>
                  Given audio of a DX7 preset, and a gallery of DX7 presets that might have made that audio,
                  can a neural network figure out which preset actually made the audio?
                </p>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>What's the DX7?</h3>
                <p>
                  The Yamaha DX7 is a beloved and well-researched FM synthesizer from 1983 that defined the sound of
                  80s pop music. It uses <strong>frequency modulation</strong> (FM) synthesis, where
                  6 operators (sine oscillators) modulate each other according to one of 32 algorithms.
                  Each <span className={twClasses.presetText}>preset</span> point here is a DX7 patch
                  (145 parameters); each <span className={twClasses.audioText}>audio</span> point is what
                  it sounds like.
                </p>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>How It Works</h3>
                <p>
                  Two neural networks convert audio clips and preset parameters into
                  <strong> embeddings</strong>—lists of numbers where similar things produce similar
                  patterns. Audio goes through a convolutional network; presets go through a{' '}
                  <strong>graph neural network (DX7-GNN)</strong> whose message passing follows the
                  preset's FM routing, like modulation signals flowing through the synth. Both are
                  trained together with a non-contrastive objective (SLAP) so audio and preset
                  embeddings live in the same "space," making them directly comparable.
                </p>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>Never-Seen Topologies</h3>
                <p>
                  Every point in this demo uses one of eight DX7 algorithms (4, 8, &hellip;, 32)
                  that were <strong>completely held out of training</strong>. Because the DX7-GNN
                  shares its weights across all operators and layers, it can encode signal routings
                  it has never seen.
                </p>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>The Visualization</h3>
                <p>
                  The scatter plot projects these high-dimensional embeddings down to 2D using t-SNE.
                  <strong> This is just for visualization</strong>—actual similarity is computed in
                  the original space. Click any point to hear it and see its DX7 parameters.
                </p>
              </section>

              <section className={`${twClasses.infoHighlightBg} border ${twClasses.infoHighlightBorder} rounded-lg p-4`}>
                <h3 className={`text-lg font-medium ${twClasses.infoHighlightTitle} mb-2`}>
                  Cross-Modal Retrieval
                </h3>
                <p>
                  The key test: select an <span className={twClasses.audioText}>audio</span> point—can
                  the model find its matching <span className={twClasses.presetText}>preset</span>? The
                  ground truth pair is marked with <span className={twClasses.groundTruthMarker}>*</span> in
                  the neighbors list. On these never-seen topologies, the DX7-GNN ranks the exact
                  preset first among all 4,096 candidates 52% of the time, and in the top ten 88%
                  of the time.
                </p>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>Why It Matters</h3>
                <p>
                  This enables <strong>sound matching</strong>: hear a sound you like,
                  find the synthesizer preset that creates it. It could power searchable preset libraries,
                  help producers recreate classic sounds, or enable AI-assisted sound design.
                </p>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>Retrieval Modes</h3>
                <ul className="list-disc list-inside space-y-1">
                  <li><strong>Cross:</strong> Find neighbors of the opposite type (audio → preset or preset → audio)</li>
                  <li><strong>Intra:</strong> Find neighbors of the same type (audio → audio or preset → preset)</li>
                </ul>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>The Research</h3>
                <p>
                  This demo accompanies the DAFx 2026 paper <em>FM Synthesizer Audio-Parameter Shared
                  Embeddings</em> (FM-SynAPSE) by David Braun and Adam Finkelstein, Princeton
                  University. Code and model weights are available at{' '}
                  <a href="https://github.com/DBraun/SynAPSE" className={twClasses.audioText + ' underline'} target="_blank" rel="noopener noreferrer">
                    github.com/DBraun/SynAPSE
                  </a>.
                </p>
              </section>

              <section>
                <h3 className={`text-lg font-medium ${twClasses.textPrimary} mb-2`}>Acknowledgments</h3>
                <p>
                  DX7 FM synthesis engine is a TypeScript port of{' '}
                  <a href="https://github.com/mmontag/dx7-synth-js" className={twClasses.audioText + ' underline'} target="_blank" rel="noopener noreferrer">
                    dx7-synth-js
                  </a>{' '}
                  by Matt Montag. Copyright &copy; 2014 Matt Montag.{' '}
                  <a href="https://raw.githubusercontent.com/mmontag/dx7-synth-js/refs/heads/master/LICENSE.txt" className={twClasses.audioText + ' underline'} target="_blank" rel="noopener noreferrer">
                    MIT License
                  </a>.
                </p>
              </section>
            </div>

            <div className={`px-6 py-4 border-t ${twClasses.borderSubtle} flex justify-center`}>
              <button
                onClick={() => setIsOpen(false)}
                className={`px-12 py-3 ${twClasses.buttonPrimary} rounded-lg text-lg font-medium`}
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
