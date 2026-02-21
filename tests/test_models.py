"""Tests for model architecture (shapes and forward pass)."""

import torch
import pytest
from showdown.models.embeddings import PokemonEncoder, TeamEncoder
from showdown.models.win_predictor import WinPredictor


BATCH = 4
TEAM_SIZE = 6
NUM_SPECIES = 100
NUM_MOVES = 200
NUM_ITEMS = 50
NUM_ABILITIES = 80


class TestPokemonEncoder:
    def setup_method(self):
        self.encoder = PokemonEncoder(
            num_species=NUM_SPECIES,
            num_moves=NUM_MOVES,
            num_items=NUM_ITEMS,
            num_abilities=NUM_ABILITIES,
            output_dim=64,
        )

    def test_output_shape(self):
        species = torch.randint(0, NUM_SPECIES, (BATCH, TEAM_SIZE))
        moves = torch.randint(0, NUM_MOVES, (BATCH, TEAM_SIZE, 4))
        items = torch.randint(0, NUM_ITEMS, (BATCH, TEAM_SIZE))
        abilities = torch.randint(0, NUM_ABILITIES, (BATCH, TEAM_SIZE))

        out = self.encoder(species, moves, items, abilities)
        assert out.shape == (BATCH, TEAM_SIZE, 64)

    def test_padding_handling(self):
        species = torch.zeros(BATCH, TEAM_SIZE, dtype=torch.long)
        moves = torch.zeros(BATCH, TEAM_SIZE, 4, dtype=torch.long)
        items = torch.zeros(BATCH, TEAM_SIZE, dtype=torch.long)
        abilities = torch.zeros(BATCH, TEAM_SIZE, dtype=torch.long)

        out = self.encoder(species, moves, items, abilities)
        assert out.shape == (BATCH, TEAM_SIZE, 64)
        assert not torch.isnan(out).any()


class TestTeamEncoder:
    def setup_method(self):
        self.encoder = TeamEncoder(pokemon_dim=64, team_dim=128)

    def test_output_shape(self):
        pokemon_reprs = torch.randn(BATCH, TEAM_SIZE, 64)
        out = self.encoder(pokemon_reprs)
        assert out.shape == (BATCH, 128)

    def test_with_mask(self):
        pokemon_reprs = torch.randn(BATCH, TEAM_SIZE, 64)
        mask = torch.zeros(BATCH, TEAM_SIZE, dtype=torch.bool)
        mask[:, 4:] = True  # mask last 2 slots

        out = self.encoder(pokemon_reprs, mask=mask)
        assert out.shape == (BATCH, 128)
        assert not torch.isnan(out).any()


class TestWinPredictor:
    def setup_method(self):
        self.model = WinPredictor(
            num_species=NUM_SPECIES,
            num_moves=NUM_MOVES,
            num_items=NUM_ITEMS,
            num_abilities=NUM_ABILITIES,
            pokemon_dim=64,
            team_dim=128,
        )

    def _make_inputs(self):
        t1_sp = torch.randint(1, NUM_SPECIES, (BATCH, TEAM_SIZE))
        t1_mv = torch.randint(1, NUM_MOVES, (BATCH, TEAM_SIZE, 4))
        t1_it = torch.randint(1, NUM_ITEMS, (BATCH, TEAM_SIZE))
        t1_ab = torch.randint(1, NUM_ABILITIES, (BATCH, TEAM_SIZE))
        t2_sp = torch.randint(1, NUM_SPECIES, (BATCH, TEAM_SIZE))
        t2_mv = torch.randint(1, NUM_MOVES, (BATCH, TEAM_SIZE, 4))
        t2_it = torch.randint(1, NUM_ITEMS, (BATCH, TEAM_SIZE))
        t2_ab = torch.randint(1, NUM_ABILITIES, (BATCH, TEAM_SIZE))
        return t1_sp, t1_mv, t1_it, t1_ab, t2_sp, t2_mv, t2_it, t2_ab

    def test_forward(self):
        inputs = self._make_inputs()
        out = self.model(*inputs)
        assert out.shape == (BATCH,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_with_rating_features(self):
        inputs = self._make_inputs()
        rating_features = torch.randn(BATCH, 2)
        out = self.model(*inputs, rating_features=rating_features)
        assert out.shape == (BATCH,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gradient_flows(self):
        """Ensure gradients flow through the entire model."""
        inputs = self._make_inputs()
        rating_features = torch.randn(BATCH, 2)
        out = self.model(*inputs, rating_features=rating_features)
        loss = out.sum()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
