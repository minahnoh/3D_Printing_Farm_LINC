from config_SimPy import *

class Item:
    """
    Class representing an item in the system.

    Attributes:
        id_order: ID of the order this item belongs to
        id_patient: ID of the patient this item belongs to
        id_item: ID of this item
        type_item: Type of item (e.g., aligner, retainer)
        is_completed: Flag indicating if the manufacturing of the item is completed
        is_defect: Flag indicating if the item is defective
    """

    def __init__(self, id_order, id_patient, id_item):
        self.id_order = id_order
        self.id_patient = id_patient
        self.id_item = id_item
        self.type_item = "aligner"  # default
        self.is_completed = False
        self.is_defect = False


class Patient:
    """
    Class representing a patient in the system.

    Attributes:
        id_order: ID of the order this patient belongs to
        id_patient: ID of this patient
        num_items: Number of items for this patient
        list_items: List of items for this patient
        is_completed: Flag indicating if the manufacturing of all items for this patient is completed
        item_counter: Counter for item IDs
    """

    def __init__(self, id_order, id_patient):
        """
        Create a patient with the given IDs.

        Args:
            id_order: ID of the order this patient belongs to
            id_patient: ID of this patient
        """
        self.id_order = id_order
        self.id_patient = id_patient
        self.num_items = NUM_ITEMS_PER_PATIENT()
        self.list_items = []
        self.is_completed = False
        self.item_counter = 1


        # Create items for this patient using the provided function
        self.list_items = self._create_items_for_patient(
            self.id_order, self.id_patient, self.num_items)

    def _create_items_for_patient(self, id_order, id_patient, num_items):
        """Create items for a patient"""
        items = []
        for _ in range(num_items):
            item_id = self._get_next_item_id()
            item = Item(id_order, id_patient, item_id)
            items.append(item)

            # # Debugging
            # print(f"[DEBUG] Order {id_order} - Patient {id_patient}: Created Item {item_id} with type: {item.type_item}")

        return items
      
    def _get_next_item_id(self):
        """Get next item ID and increment counter"""
        item_id = self.item_counter
        self.item_counter += 1
        return item_id

    def check_completion(self):
        """Check if all items for this patient are completed"""
        if all(item.is_completed for item in self.list_items):
            self.is_completed = True
        return self.is_completed
    

class Order:
    """
    Class representing an order in the system.

    Attributes:
        id_order: ID of this order
        num_patients: Number of patients for this order
        list_patients: List of patients for this order
        due_date: Due date of this order
        time_start: Start time of this order
        time_end: End time of this order
        patient_counter: Counter for patient IDs

    """

    def __init__(self, id_order):
        """
        Create an order with the given ID.

        Args:
            id_order: ID of this order 
        """

        self.id_order = id_order
        self.num_patients = NUM_PATIENTS_PER_ORDER()
        self.list_patients = []
        self.due_date = ORDER_DUE_DATE
        self.time_start = None
        self.time_end = None
        self.patient_counter = 1

        # Create patients for this order using the provided function
        self.list_patients = self._create_patients_for_order(
            self.id_order, self.num_patients)

    def _create_patients_for_order(self, id_order, num_patients):
        """Create patients for an order"""
        patients = []
        for _ in range(num_patients):
            patient_id = self._get_next_patient_id()
            
            patient = Patient(id_order, patient_id)
            patients.append(patient)

            ## Debugging: Print patient and item details
            # print(f"[Debug] Order {id_order}: Created Patient {patient_id} with items:")
            # for item in patient.list_items:
            #     print(f"    Item ID: {item.id_item} (Type: {item.type_item})")
        return patients

    def _get_next_patient_id(self):
        """Get next patient ID and increment counter"""
        patient_id = self.patient_counter
        self.patient_counter += 1
        return patient_id

    def check_completion(self,patient):
        """Check if all patients in this order are completed"""
        if all(patient.check_completion() for patient in self.list_patients):
            return True
        return False 


class Customer():
    """
    Class representing a customer in the system.

    Attributes:
        env: Simulation environment
        order_receiver: Order receiver object
        logger: Logger object
        order_counter: Counter for order IDs
        processing: Process for creating orders
    """

    def __init__(self, env, order_receiver, logger):
        self.env = env
        self.order_receiver = order_receiver
        self.logger = logger

        # Initialize ID counters
        self.order_counter = 1

        # Automatically start the process when the Customer is created
        self.processing = env.process(self.create_order())

    def get_next_order_id(self):
        """Get next order ID and increment counter"""
        order_id = self.order_counter
        self.order_counter += 1
        return order_id

    def create_order(self):
        """Create orders periodically"""
        while True:
            # Create a new order
            order_id = self.get_next_order_id()
            order = Order(order_id)
            order.time_start = self.env.now

            # # Log order creation
            #  self.logger.log_event(
            #     "Order", f"Created Order {order.id_order} (Patients: {order.num_patients}, Total items: {sum(len(patient.list_items) for patient in order.list_patients)})")
            
            # # Debugging
            # patient_details_str = "\n".join(
            #    f"    Patient ID: {patient.id_patient}\n" +
            #    "\n".join(f"        Item ID: {item.id_item}" for item in patient.list_items) for patient in order.list_patients)
            # print(f"check: Order {order.id_order} consists of patients and their items:\n{patient_details_str}")
            
            # Send the order
            self.send_order(order)
            
            # Wait for next order cycle
            yield self.env.timeout(CUST_ORDER_CYCLE)

    def send_order(self, order):
        """Send the order to the receiver"""
        # if self.logger:
        #    self.logger.log_event(
        #        "Order", f"Sending Order {order.id_order} to processor")
        self.order_receiver.receive_order(order)
            
class OrderReceiver:
    """Interface for order receiving objects"""
    def receive_order(self, order):
        """Method to process orders (implemented by subclasses)"""
        pass


class SimpleOrderReceiver(OrderReceiver):
    """Simple order receiver for testing"""

    def __init__(self, env, logger=None):
        self.env = env
        self.logger = logger
        self.received_orders = []

    def receive_order(self, order):
        """Receive order and log it"""
        self.received_orders.append(order)
        
        # # Debugging
        #patient_details_str = "\n".join(
        #    f"    Patient ID: {patient.id_patient}\n" +
        #    "\n".join(f"        Item ID: {item.id_item}" for item in patient.list_items) for patient in order.list_patients)
        #print(f"[Debug] Order {order.id_order} consists of patients and their items:\n{patient_details_str}")

        self.logger.log_event(
            "Order", f"OrderReceiver recevied Order {order.id_order} (Patients: {order.num_patients}, Total items: {sum(len(patient.list_items) for patient in order.list_patients)})")