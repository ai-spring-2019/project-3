"""
Tom Helmuth
This is my implementation of Dominion!
"""

import copy, random, sys

class Dominion():

    def __init__(self, p1, p2, verbose):
        self.verbose = verbose

        self.players = [p1, p2]
        self.current_player = 0
        self.turn = 0

        # Setup cards in tableau
        victory = [Estate, Duchy, Province]
        self.basic_victory = [PileOfCards(crd() * 8) for crd in victory]
        self.curses = PileOfCards(Curse() * 10)
        self.provinces = self.basic_victory[2]

        treasure = [Copper, Silver, Gold]
        self.basic_treasure = [PileOfCards(Copper() * 46),
                                PileOfCards(Silver() * 40),
                                PileOfCards(Gold() * 30)]

        kingdom = [Village, Smithy, Market, Witch, Gardens, CouncilRoom, Chapel,
                   Laboratory, Moneylender, Workshop]
        self.kingdom = [PileOfCards(crd() * 10) for crd in kingdom]
        self.kingdom.sort(key=lambda poc : poc.top().cost_and_name())

        self.all_piles = [self.curses] + self.basic_victory + self.basic_treasure + self.kingdom


    def play(self):
        """Plays an entire game of Dominion."""

        while not self.game_end():
            self.turn += 1
            self.play_turn()

            self.current_player = (self.current_player + 1) % 2

        return self.find_out_who_won()


    def play_turn(self):
        """Has current player play one turn."""

        self.log("\n               <<<<<<<<<< TURN {}: PLAYER {}'S TURN >>>>>>>>>".format(self.turn, self.current_player))

        self.print_cards()

        actions = 1
        buys = 1
        coins = 0

        player = self.players[self.current_player]

        self.log_basics(actions, buys, coins)
        self.log()
        self.log("Played cards")
        self.log("------------")
        self.log(player.play_area)
        self.log()
        self.log("Hand")
        self.log("----")
        self.log(player.hand)

        #### Phase 1: Actions
        while actions > 0 and player.has_actions():

            # Choose an action card to play.
            action_to_play = player.choose_action_to_play()
            self.log()
            self.log("Playing:", action_to_play)

            actions -= 1

            # Play the card:
            actions += action_to_play.plus_action
            buys += action_to_play.plus_buy
            coins += action_to_play.plus_coin

            for _ in range(action_to_play.plus_card):
                card = player.draw_card()
                if card == None:
                    self.log("Can't draw; deck and discard empty.")
                else:
                    self.log("Draw:", card)


            self.log()
            # Specific card activities
            if action_to_play.name == "Moneylender":
                if player.hand.contains("Copper"):
                    # Trash copper, get 3 coins
                    self.log("Moneylender trashes a Copper and gains $3")
                    player.hand.trash("Copper")
                    coins += 3

            if action_to_play.name == "Chapel":
                # Trash up to 4 cards in this order: Curse, Estate, Copper
                for _ in range(4):
                    if player.hand.contains("Curse"):
                        player.hand.trash("Curse")
                        self.log("Chapel trashes a Curse.")

                    elif player.hand.contains("Estate"):
                        player.hand.trash("Estate")
                        self.log("Chapel trashes an Estate.")

                    elif player.hand.contains("Copper"):
                        player.hand.trash("Copper")
                        self.log("Chapel trashes a Copper.")

                    else:
                        break

            if action_to_play.name == "Witch":
                # Check if any Curses left:
                if self.can_buy("Curse", 0):
                    other_player = (self.current_player + 1) % 2
                    self.players[other_player].gain_card(self.buy("Curse", 0))
                    self.log("Player {} gains a Curse from Witch.".format(other_player))

            if action_to_play.name == "CouncilRoom":
                # Other player should draw
                other_player = (self.current_player + 1) % 2
                self.players[other_player].draw_card()
                self.log("Player {} draws a card from CouncilRoom.".format(other_player))

            if action_to_play.name == "Workshop":
                # Can gain a card costing up to $4
                done_gaining = False
                while not done_gaining:
                    self.log('Player {}, what card would you like to gain from Workshop? Or enter "pass" to not gain.'.format(self.current_player))
                    card_name = player.choose_gain(self, 4, 1)

                    # Check if passing
                    if card_name.lower() == "pass":
                        break

                    # Check if you can buy it
                    if not self.can_buy(card_name, 4):
                        self.log("Sorry, you cannot gain {}, try again.".format(card_name))
                        continue

                    card = self.buy(card_name, 4)
                    self.log("Player {} bought {}".format(self.current_player, card))
                    player.gain_card(card)

                    done_gaining = True

            self.log()
            self.log_basics(actions, buys, coins)
            self.log()
            self.log("Played cards")
            self.log("------------")
            self.log(player.play_area)
            self.log()
            self.log("Hand")
            self.log("----")
            self.log(player.hand)

        #### Phase 2: Buy
        # Play all treasures
        self.log_basics(actions, buys, coins)
        self.log("####################           Buy phase           ####################")

        for card in player.hand.cards:
            if "treasure" in card.type:
                coins += card.plus_coin
                self.log("Playing: {} for ${}".format(card, card.plus_coin))

        # Time for player to decide what card to buy
        while buys > 0:
            self.log()
            self.print_cards()
            self.log_basics(actions, buys, coins)
            self.log()
            self.log('Player {}, what card would you like to buy? Or enter "pass" to not buy.'.format(self.current_player))
            card_name = player.choose_gain(self, coins, buys)

            # Check if passing
            if card_name.lower() == "pass":
                break

            # Check if you can buy it
            if not self.can_buy(card_name, coins):
                self.log("Sorry, you cannot buy {}, try again.".format(card_name))
                continue

            card = self.buy(card_name, coins)
            self.log("Player {} bought {}".format(self.current_player, card))

            # Decrement buys and coins, add card to player's discard
            buys -= 1
            coins -= card.cost
            player.gain_card(card)

        #### Phase 3: Cleanup
        player.cleanup()



    def can_buy(self, card_name, coins):
        """Checks if you could buy card_name with given coins. Will be false
        for any of these reasons:
            - no pile with that name
            - not enough coins to buy a card with that name
            - that pile is out of cards"""

        for pile in self.all_piles:
            if pile.name.lower() == card_name.lower() and pile.size() > 0 and pile.cost <= coins:
                return True

        return False

    def buy(self, card_name, coins):
        """Buys card of given name, if you can"""

        for pile in self.all_piles:
            if pile.name.lower() == card_name.lower() and pile.size() > 0 and pile.cost <= coins:
                return pile.draw()


    def game_end(self):
        """True if the game has ended."""

        if self.turn >= 100:
            self.log("\n\nGAME ENDS BECAUSE 100 TURNS HAVE BEEN PLAYED!")
            return True

        if self.provinces.empty():
            self.log("\n\nGAME ENDS BECAUSE PROVINCE PILE IS EMPTY!")
            return True

        empty_piles = 0
        for pile in self.all_piles:
            if pile.empty():
                empty_piles += 1

        if empty_piles >= 3:
            self.log("\n\nGAME ENDS BECAUSE 3 PILES ARE EMPTY!")
            return True

        return False

    def find_out_who_won(self):
        """Prints end of game report, and returns which player won."""

        p0_score = self.score_one_player(self.players[0], 0)
        p1_score = self.score_one_player(self.players[1], 1)

        self.log()

        if p0_score > p1_score:
            self.log("Player 0 wins!")
            return 0

        if p0_score < p1_score:
            self.log("Player 1 wins!")
            return 1

        if self.current_player == 1:
            self.log("Player 1 wins!")
            return 1

        self.log("Players rejoice in a shared victory!")
        return "draw"


    def score_one_player(self, player, num):
        """Finds score for one player and prints end of game info."""
        player.consolodate_cards_and_sort()

        self.log()
        self.log("Player {}:".format(num))
        self.log("Cards:", player.draw)

        # Calculate score
        score = 0
        provinces = 0
        duchies = 0
        estates = 0
        gardens = 0
        curses = 0
        num_cards = len(player.draw.cards)

        for card in player.draw.cards:
            score += card.vp
            if card.name == "Province":
                provinces += 1
            if card.name == "Duchy":
                duchies += 1
            if card.name == "Estate":
                estates += 1
            if card.name == "Gardens":
                gardens += 1
                score += (num_cards // 10)
            if card.name == "Curse":
                curses += 1


        self.log()
        self.log("{} Provinces for {} VP".format(provinces, provinces * 6))
        self.log("{} Duchies for   {} VP".format(duchies, duchies * 3))
        self.log("{} Estates for   {} VP".format(estates, estates * 1))
        self.log("{} Gardens for   {} VP (because of {} cards in deck)".format(gardens, gardens * (num_cards // 10), num_cards))
        self.log("{} Curses for    {} VP".format(curses, curses * -1))

        self.log("Score:", score)

        return score

    def print_cards(self):
        """Prints the tableau"""

        self.log("-" * 80)

        self.print_width("Victory Cards")
        self.print_width("-------------")
        vp = ""
        for v in self.basic_victory:
            vp += "%-25s" % str(v)
        self.print_width(vp)
        self.print_width(self.curses)

        self.print_width()
        self.print_width("Treasure Cards")
        self.print_width("-------------")
        tre = ""
        for t in self.basic_treasure:
            tre += "%-25s" % str(t)
        self.print_width(tre)

        self.print_width()
        self.print_width("Kingdom Cards")
        self.print_width("-------------")

        for card_cost in range(20):
            cards_at_cost = filter(lambda poc: poc.cost == card_cost, self.kingdom)
            cds = ""
            for i, c in enumerate(cards_at_cost):
                cds += "%-25s" % str(c)
                if i % 3 == 2:
                    self.print_width(cds)
                    cds = ""
            if cds != "":
                self.print_width(cds)

        self.log("-" * 80)

    def print_width(self, string=""):
        """Prints a line with |  | on either side, making the whole printed line
        at least 80 characters wide."""

        self.log("| %-76s |" % string)

    def log(self, *args):
        """Prints *args if verbose=True"""
        if self.verbose:
            print(*args)

    def log_basics(self, actions, buys, coins):
        self.log("\n#######################################################################")
        self.log("##############   Player: {}   Actions: {}   Buys: {}   ${}   ##############".format(self.current_player, actions, buys, coins))


################################################################################

class PileOfCards():

    def __init__(self, cards=None):
        if cards == None:
            cards = []

        self.cards = cards # The cards in this pile
        if len(self.cards) != 0:
            card = self.cards[0]
            self.name = card.name
            self.cost = card.cost
            self.basic = card.basic

    def size(self):
        return len(self.cards)

    def empty(self):
        return len(self.cards) == 0

    def top(self):
        return self.cards[0]

    def __str__(self):
        return "$%i %s (%i)" % (self.cost, self.name, self.size())

    def __add__(self, other):
        """Concatenates two piles of cards, returning a new pile."""

        if not isinstance(other, PileOfCards):
            raise TypeError("Can't + a PileOfCards and something that is not a PileOfCards")
        return PileOfCards(self.cards + other.cards)

    def draw(self):
        """Draws one card from this pile"""
        if self.size() > 0:
            return self.cards.pop()
        return

    def add(self, card):
        """Adds card to this pile"""
        self.cards.append(card)

    def add_all(self, cards):
        """Adds cards to this pile"""
        self.cards.extend(cards)

    def return_all(self):
        """Returns all cards and sets cards to empty."""
        cards = self.cards
        self.cards = []
        return cards

    def shuffle(self):
        random.shuffle(self.cards)


class PileOfDifferentCards(PileOfCards):
    """Any pile of cards that isn't all the same card."""

    def __init__(self, cards=None):
        PileOfCards.__init__(self, cards)

    def __str__(self):
        if self.empty():
            return ""

        result = str(self.cards[0])
        for card in self.cards[1:]:
            result += "   " + str(card)

        return result

class DrawDeck(PileOfDifferentCards):
    """The draw deck for a player"""

    def __init__(self, cards=None):
        PileOfCards.__init__(self, cards)

class DiscardPile(PileOfDifferentCards):
    """The discard pile for a player"""

    def __init__(self, cards=None):
        PileOfCards.__init__(self, cards)

class PlayArea(PileOfDifferentCards):
    """The play area for a player."""

    def __init__(self, cards=None):
        PileOfCards.__init__(self, cards)

class Hand(PileOfDifferentCards):
    """The hand for a player"""

    def __init__(self, cards=None):
        PileOfCards.__init__(self, cards)

    def contains(self, card_name):
        """True if card_name is in hand, False otherwise."""
        for i, card in enumerate(self.cards):
            if card.name == card_name:
                return True
        return False

    def trash(self, card_name):
        """Trashes one of card_name."""
        for i, card in enumerate(self.cards):
            if card.name == card_name:
                self.cards.pop(i)
                return


################################################################################

class Card():

    def __init__(self):
        self.name = ""
        self.type = [] # Options: action, treasure, victory, curse, (attack, reaction, ...)
        self.basic = False # True if a basic victory or treasure, False if Kingdom card
        self.cost = 0
        self.vp = 0
        self.priority = -1

        # These tell what you get when you play an action.
        self.plus_action = 0
        self.plus_buy = 0
        self.plus_coin = 0
        self.plus_card = 0

    def __str__(self):
        return self.name

    def cost_and_name(self):
        return "$%i %s" % (self.cost, self.name)

    def __mul__(self, num):
        """Must take num as an integer argument. Returns a list of num copies of this card."""

        if not isinstance(num, int):
            raise TypeError("Can't multiply a card by anything besides an int.")

        cards = []
        for _ in range(num):
            cards.append(copy.deepcopy(self))
        return cards

################################################################################

class Estate(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Estate"
        self.type = ["victory"]
        self.basic = True
        self.cost = 2
        self.vp = 1

class Duchy(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Duchy"
        self.type = ["victory"]
        self.basic = True
        self.cost = 5
        self.vp = 3

class Province(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Province"
        self.type = ["victory"]
        self.basic = True
        self.cost = 8
        self.vp = 6

class Curse(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Curse"
        self.type = ["curse"]
        self.basic = True
        self.cost = 0
        self.vp = -1

class Copper(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Copper"
        self.type = ["treasure"]
        self.basic = True
        self.cost = 0
        self.plus_coin = 1

class Silver(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Silver"
        self.type = ["treasure"]
        self.basic = True
        self.cost = 3
        self.plus_coin = 2

class Gold(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Gold"
        self.type = ["treasure"]
        self.basic = True
        self.cost = 6
        self.plus_coin = 3


################################################################################
# Market, Witch, Gardens, CouncilRoom, Chapel,
#           Laboratory, Moneylender, Workshop

class Village(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Village"
        self.type = ["action"]
        self.basic = False
        self.cost = 3
        self.priority = 10

        self.plus_action = 2
        self.plus_card = 1

class Market(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Market"
        self.type = ["action"]
        self.basic = False
        self.cost = 5
        self.priority = 9

        self.plus_action = 1
        self.plus_buy = 1
        self.plus_coin = 1
        self.plus_card = 1

class Laboratory(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Laboratory"
        self.type = ["action"]
        self.basic = False
        self.cost = 5
        self.priority = 8

        self.plus_action = 1
        self.plus_card = 2

class Witch(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Witch"
        self.type = ["action", "attack"]
        self.basic = False
        self.cost = 5
        self.priority = 7

        self.plus_card = 2

class Smithy(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Smithy"
        self.type = ["action"]
        self.basic = False
        self.cost = 4
        self.priority = 6

        self.plus_card = 3

class CouncilRoom(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "CouncilRoom"
        self.type = ["action"]
        self.basic = False
        self.cost = 5
        self.priority = 5

        self.plus_card = 4
        self.plus_buy = 1

class Workshop(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Workshop"
        self.type = ["action"]
        self.basic = False
        self.cost = 3
        self.priority = 4

class Moneylender(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Moneylender"
        self.type = ["action"]
        self.basic = False
        self.cost = 4
        self.priority = 3

class Chapel(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Chapel"
        self.type = ["action"]
        self.basic = False
        self.cost = 2
        self.priority = 2

class Gardens(Card):

    def __init__(self):
        Card.__init__(self)
        self.name = "Gardens"
        self.type = ["victory"]
        self.basic = False
        self.cost = 4


################################################################################

class Player():

    def __init__(self):
        self.draw = DrawDeck((Estate() * 3) + (Copper() * 7))
        self.discard = DiscardPile()
        self.hand = Hand()
        self.play_area = PlayArea()
        self.cards_owned = {"Estate": 3, "Copper": 7}

        self.shuffle()
        for _ in range(5):
            self.hand.add(self.draw.draw())

    def __str__(self):
        return """PLAYER:
  |- Draw:      {}
  |- Discard:   {}
  |- Hand:      {}
  |- Play Area: {}
""".format(self.draw, self.discard, self.hand, self.play_area)

    def shuffle(self):
        self.draw.shuffle()

    def draw_card(self):
        """Draws a single card from draw to hand. If draw is empty, reshuffles
        discard and then draws. If discard is also empty, doesn't draw anything."""
        if not self.draw.empty():
            card = self.draw.draw()
            self.hand.add(card)
            return card

        # If here, draw deck is empty
        if self.discard.empty():
            # Can't draw anything if draw deck and discard are empty
            return

        # Reshuffle discard pile into draw deck
        self.draw.add_all(self.discard.return_all())
        self.shuffle()

        # Draw a card
        card = self.draw.draw()
        self.hand.add(card)
        return card

    def has_actions(self):
        """True if the player has an action card in hand."""
        for card in self.hand.cards:
            if "action" in card.type:
                return True
        return False

    def gain_card(self, card):
        """Gains this card to discard pile."""
        self.discard.add(card)
        previous_number = self.cards_owned.get(card.name, 0)
        self.cards_owned[card.name] = previous_number + 1

    def choose_action_to_play(self):
        """Picks which action card to play based on simple priority order.
        Should only be called if has_actions is true"""

        card_index = -1
        priority = -1
        for i, card in enumerate(self.hand.cards):
            if "action" in card.type and card.priority > priority:
                card_index = i
                priority = card.priority

        # Remove card from hand, add to in play
        action_card = self.hand.cards.pop(card_index)
        self.play_area.add(action_card)

        return action_card

    def cleanup(self):
        """Performs cleanup at end of turn."""

        # Add play_area and hand to discard
        self.discard.add_all(self.play_area.return_all())
        self.discard.add_all(self.hand.return_all())

        # Draw 5 cards
        for _ in range(5):
            self.draw_card()

    def consolodate_cards_and_sort(self):
        """Consolodates all cards into draw deck for end of game scoring."""
        self.draw.add_all(self.hand.return_all())
        self.draw.add_all(self.discard.return_all())
        self.draw.add_all(self.play_area.return_all())

        self.draw.cards.sort(key=sort_key_for_end_game, reverse=True)

def sort_key_for_end_game(card):
    """Returns a key for sorting in the end game scoring.
    Super hacky, but puts victory cards first."""
    if "victory" in card.type:
        return "victory" + card.cost_and_name()
    if "treasure" in card.type:
        return "treasure" + card.cost_and_name()
    return "action" + card.cost_and_name()


class HumanPlayer(Player):
    """Allows a human to play."""

    def __init__(self):
        Player.__init__(self)

    def choose_gain(self, game, coins, buys):
        """Returns string of the card would like to gain.
        Takes the Dominion object as "game" since you may need the can_buy
        method to determine if you can buy a card."""
        card = input("> ")
        return card


class BigMoney(Player):
    """Implements a simple big-money strategy, selecting cards in this order:
       - Province
       - Gold
       - Smithy (at most 2)
       - Silver"""

    def __init__(self):
        Player.__init__(self)
        self.smithy_count = 0

    def choose_gain(self, game, coins, buys):
        """Uses simple rules above"""

        if game.can_buy("Province", coins):
            return "Province"
        if game.can_buy("Gold", coins):
            return "Gold"
        if self.smithy_count < 2 and game.can_buy("Smithy", coins):
            self.smithy_count += 1
            return "Smithy"
        if game.can_buy("Silver", coins):
            return "Silver"
        return "pass"


################################################################################

def main():
    player0 = HumanPlayer()
    player1 = HumanPlayer()
    #player1 = BigMoney()

    verbose = True

    game = Dominion(player0, player1, verbose)
    winner = game.play()

    if winner == "draw":
        print("Draw")
    else:
        print("Player {} won!".format(winner))


if __name__ == "__main__":
    main()
