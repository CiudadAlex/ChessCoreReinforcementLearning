package org.leviatan.chess.engine.intel.reinforcementlearning;

import java.io.IOException;
import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.engine.CPUPlayer;
import org.leviatan.chess.engine.deeplearning.ConfiguracionDeepLearning;
import org.leviatan.chess.engine.intel.generadorarbol.HelperMovimientosPosibles;
import org.leviatan.chess.engine.intel.reinforcementlearning.network.ThreatEvaluationNetwork;
import org.leviatan.chess.ui.UserIntefaceInteractor;

/**
 * CPUPlayerReinforcementLearningImpl.
 *
 * @author Alejandro
 *
 */
public class CPUPlayerReinforcementLearningImpl implements CPUPlayer {

    private final ThreatEvaluationNetwork threatEvaluationNetwork;

    /**
     * Constructor for CPUPlayerReinforcementLearningImpl.
     *
     * @throws IOException
     */
    public CPUPlayerReinforcementLearningImpl() throws IOException {

        final String pathDirStoreLoad = ConfiguracionDeepLearning.DIR_MODELOS;
        this.threatEvaluationNetwork = new ThreatEvaluationNetwork(pathDirStoreLoad);
    }

    @Override
    public Tablero realizarJugadaCPU(final Tablero tablero, final UserIntefaceInteractor userIntefaceInteractor, final Bando bandoCPU) {

        Movimiento mejorMovimiento = null;
        double mejorScoreMovimiento = Double.NEGATIVE_INFINITY;

        final List<Movimiento> listaMovimientosPosiblesBando = HelperMovimientosPosibles.getMovimientosPosiblesDeBando(tablero, bandoCPU);

        for (final Movimiento movimiento : listaMovimientosPosiblesBando) {

            final Tablero tableroMovido = tablero.clonar();
            tableroMovido.realizarMovimiento(movimiento);

            final double threatCPU = this.threatEvaluationNetwork.getThreatOfGivenBando(tableroMovido, bandoCPU);
            final double threatHuman = this.threatEvaluationNetwork.getThreatOfGivenBando(tableroMovido, bandoCPU.getBandoOpuesto());

            final double score = threatCPU - threatHuman;

            if (mejorScoreMovimiento < score) {
                mejorScoreMovimiento = score;
                mejorMovimiento = movimiento;
            }
        }

        tablero.realizarMovimiento(mejorMovimiento);
        return tablero;
    }

}
