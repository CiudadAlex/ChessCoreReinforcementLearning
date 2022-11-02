package org.leviatan.chess.engine.intel.reinforcementlearning;

import org.leviatan.chess.engine.deeplearning.ConfiguracionDeepLearning;
import org.leviatan.chess.engine.intel.reinforcementlearning.training.TrainerThreat;

/**
 * TestTrainerThreat.
 *
 * @author Alejandro
 *
 */
public final class TestTrainerThreat {

    private TestTrainerThreat() {
    }

    /**
     * Main method.
     *
     * @param args
     *            args
     * @throws Exception
     */
    public static void main(final String[] args) throws Exception {
        TrainerThreat.trainListPartidasDeFichero(ConfiguracionDeepLearning.DIR_MODELOS, 0, true);
    }


}
