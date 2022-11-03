package org.leviatan.chess.engine.intel.reinforcementlearning.network;

import java.util.ArrayList;
import java.util.List;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Movimiento;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.board.TipoFicha;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.intel.Inteligencia;
import org.leviatan.chess.engine.intel.dto.MovimientoYConsecuencias;
import org.leviatan.chess.tools.platform.AppLogger;

/**
 * ThreatEvaluationShortTerm.
 *
 * @author Alejandro
 *
 */
public class ThreatEvaluationShortTerm implements ThreatEvaluation {

    private static final double ATTENUATION_FACTOR = 0.01;

    @Override
    public List<TableroAndThreat> evaluateThreatsOfPartida(final Partida partida) {

        final Tablero tablero = new Tablero();

        final List<Movimiento> listMovimiento = partida.getListMovimiento();
        final int numMovimientos = listMovimiento.size();
        AppLogger.logDebug("Numero movimientos de esta partida: " + numMovimientos);
        int countMovimientos = 0;

        final List<TableroAndThreat> listTableroAndThreat = new ArrayList<TableroAndThreat>();

        for (final Movimiento movimiento : listMovimiento) {

            final TableroAndThreat tableroAndThreat = evaluateThreatShortTerm(tablero);
            listTableroAndThreat.add(tableroAndThreat);

            tablero.realizarMovimiento(movimiento);

            countMovimientos++;

            AppLogger.logDebug("Movimientos entrenados: " + countMovimientos + " / " + numMovimientos);
        }

        return listTableroAndThreat;
    }

    private TableroAndThreat evaluateThreatShortTerm(final Tablero tableroArg) {

        final Tablero tablero = tableroArg.clonar();

        final double threatBlancas = getThreatOfGivenBando(tablero, Bando.BLANCO);
        final double threatNegras = getThreatOfGivenBando(tablero, Bando.NEGRO);

        return new TableroAndThreat(tablero, threatBlancas, threatNegras);
    }

    public double getThreatOfGivenBando(final Tablero tablero, final Bando bando) {

        if (Bando.BLANCO.equals(bando)) {
            return getThreat(tablero);
        } else {
            return getThreat(tablero.creaTableroEspecular());
        }
    }

    private double getThreat(final Tablero tablero) {

        final MovimientoYConsecuencias movimientoYConsecuencias = Inteligencia.decideMovimientoARealizar(tablero, 2);
        final double gananciaAcumulada = movimientoYConsecuencias.getGananciaAcumulada();

        double threat = gananciaAcumulada / TipoFicha.REY.getValor();
        threat = threat * ATTENUATION_FACTOR;
        threat = threat > 1 ? 1 : threat;

        return threat;
    }
}
